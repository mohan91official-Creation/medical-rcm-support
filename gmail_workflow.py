"""Consent-driven Gmail inbox adapter for the secure RCM support pipeline.

The adapter intentionally processes only unread messages whose subject begins with
the configured test prefix. It never logs message content and sends a reply only
after both the application's HITL gate and a separate send confirmation succeed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
from email import message_from_bytes
from email.message import EmailMessage, Message
from email.utils import parseaddr
import html
import logging
from pathlib import Path
import re
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import EmailInput, Settings, SupportApplication, TicketResult, configure_runtime


ROOT = Path(__file__).resolve().parent
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
ADDRESS_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class GmailSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    gmail_oauth_client_file: Path = ROOT / "gmail-oauth-client.json"
    gmail_token_file: Path = ROOT / "gmail-token.json"
    gmail_support_address: str = ""
    gmail_subject_prefix: str = "[RCM TEST]"
    gmail_max_messages: int = Field(default=3, ge=1, le=10)
    gmail_processed_label: str = "RCM-Processed"
    gmail_rejected_label: str = "RCM-Rejected"


class GmailEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    thread_id: str
    internet_message_id: str = ""
    references: str = ""
    sender: str
    subject: str
    received_at: datetime
    content: str = Field(min_length=1, max_length=20_000)


def _safe_header(value: str, limit: int) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())[:limit]


def _decode_websafe(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _plain_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return str(payload or "")
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _html_to_text(value: str) -> str:
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p\s*>", "\n", value)
    return html.unescape(re.sub(r"(?s)<[^>]+>", " ", value))


def extract_message_body(message: Message) -> str:
    plain_parts: list[str] = []
    html_parts: list[str] = []
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_plain_text(part))
        elif content_type == "text/html":
            html_parts.append(_html_to_text(_plain_text(part)))
    body = "\n".join(plain_parts or html_parts)
    body = body.replace("\x00", " ").strip()
    if not body:
        raise ValueError("Message has no supported text body")
    return body[:20_000]


def is_automated_message(message: Message) -> bool:
    auto_submitted = (message.get("Auto-Submitted") or "").lower()
    precedence = (message.get("Precedence") or "").lower()
    sender = parseaddr(message.get("From") or "")[1].lower()
    local_part = sender.partition("@")[0]
    return (
        auto_submitted not in {"", "no"}
        or precedence in {"bulk", "junk", "list"}
        or bool(message.get("List-Unsubscribe"))
        or local_part in {"mailer-daemon", "postmaster", "no-reply", "noreply"}
    )


class GmailGateway:
    def __init__(self, settings: GmailSettings) -> None:
        self.settings = settings
        self.service: Any = None

    def authenticate(self) -> None:
        client_file = self.settings.gmail_oauth_client_file
        token_file = self.settings.gmail_token_file
        if not client_file.is_absolute():
            client_file = ROOT / client_file
        if not token_file.is_absolute():
            token_file = ROOT / token_file
        if not client_file.exists():
            raise FileNotFoundError("Gmail OAuth client file is missing")

        credentials: Credentials | None = None
        if token_file.exists():
            loaded_credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)
            if not isinstance(loaded_credentials, Credentials):
                raise TypeError("Gmail token did not contain user OAuth credentials")
            credentials = loaded_credentials
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
                authorized_credentials = flow.run_local_server(port=0, open_browser=True)
                if not isinstance(authorized_credentials, Credentials):
                    raise TypeError("Gmail OAuth flow did not return user credentials")
                credentials = authorized_credentials
            if credentials is None:
                raise RuntimeError("Gmail OAuth credentials were not created")
            token_file.write_text(credentials.to_json(), encoding="utf-8")
        self.service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    def verify_account(self) -> str:
        profile = self.service.users().getProfile(userId="me").execute()
        actual = str(profile.get("emailAddress", "")).lower()
        expected = self.settings.gmail_support_address.strip().lower()
        if not expected:
            raise ValueError("GMAIL_SUPPORT_ADDRESS must be configured")
        if actual != expected:
            raise PermissionError("Authorized Gmail account does not match GMAIL_SUPPORT_ADDRESS")
        return actual

    def unread_message_ids(self) -> list[str]:
        prefix = self.settings.gmail_subject_prefix.replace('"', "")
        query = f'is:unread in:inbox subject:"{prefix}" -from:me'
        response = (
            self.service.users()
            .messages()
            .list(userId="me", q=query, maxResults=self.settings.gmail_max_messages)
            .execute()
        )
        return [str(item["id"]) for item in response.get("messages", [])]

    def read_envelope(self, message_id: str, support_address: str) -> GmailEnvelope:
        resource = (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format="raw")
            .execute()
        )
        parsed = message_from_bytes(_decode_websafe(str(resource["raw"])))
        sender = parseaddr(parsed.get("From") or "")[1].lower()
        subject = _safe_header(parsed.get("Subject") or "", 300)
        if not ADDRESS_PATTERN.fullmatch(sender) or sender == support_address:
            raise ValueError("Message sender is not eligible for reply")
        if is_automated_message(parsed):
            raise ValueError("Automated or list message is not eligible for reply")
        if not subject.lower().startswith(self.settings.gmail_subject_prefix.lower()):
            raise ValueError("Message does not have the required test subject prefix")
        internal_date = int(resource.get("internalDate", "0")) / 1000
        received_at = datetime.fromtimestamp(internal_date, tz=timezone.utc)
        return GmailEnvelope(
            message_id=message_id,
            thread_id=str(resource["threadId"]),
            internet_message_id=_safe_header(parsed.get("Message-ID") or "", 998),
            references=_safe_header(parsed.get("References") or "", 1800),
            sender=sender,
            subject=subject,
            received_at=received_at,
            content=extract_message_body(parsed),
        )

    def send_reply(self, envelope: GmailEnvelope, reply: str) -> str:
        message = EmailMessage()
        message["To"] = envelope.sender
        message["From"] = self.settings.gmail_support_address
        message["Subject"] = (
            envelope.subject if envelope.subject.lower().startswith("re:") else f"Re: {envelope.subject}"
        )
        if envelope.internet_message_id:
            message["In-Reply-To"] = envelope.internet_message_id
            references = f"{envelope.references} {envelope.internet_message_id}".strip()
            message["References"] = references
        message.set_content(reply)
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        response = (
            self.service.users()
            .messages()
            .send(userId="me", body={"raw": encoded, "threadId": envelope.thread_id})
            .execute()
        )
        return str(response["id"])

    def label_message(self, message_id: str, label_name: str, mark_read: bool = True) -> None:
        label_id = self._ensure_label(label_name)
        body: dict[str, list[str]] = {"addLabelIds": [label_id]}
        if mark_read:
            body["removeLabelIds"] = ["UNREAD"]
        self.service.users().messages().modify(userId="me", id=message_id, body=body).execute()

    def _ensure_label(self, label_name: str) -> str:
        response = self.service.users().labels().list(userId="me").execute()
        for label in response.get("labels", []):
            if label.get("name") == label_name:
                return str(label["id"])
        created = (
            self.service.users()
            .labels()
            .create(
                userId="me",
                body={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            .execute()
        )
        return str(created["id"])


async def process_inbox(dry_run: bool) -> int:
    gmail_settings = GmailSettings()
    app_settings = Settings()
    if app_settings.auto_approve:
        raise ValueError("Set AUTO_APPROVE=false before using the Gmail workflow")
    configure_runtime(app_settings)
    gateway = GmailGateway(gmail_settings)
    await asyncio.to_thread(gateway.authenticate)
    support_address = await asyncio.to_thread(gateway.verify_account)
    message_ids = await asyncio.to_thread(gateway.unread_message_ids)
    if not message_ids:
        print(f'No unread messages found with subject prefix "{gmail_settings.gmail_subject_prefix}".')
        return 0

    support_app = SupportApplication(app_settings)
    sent = 0
    for message_id in message_ids:
        try:
            envelope = await asyncio.to_thread(gateway.read_envelope, message_id, support_address)
            payload = EmailInput(
                subject=envelope.subject,
                sender=envelope.sender,
                time=envelope.received_at,
                content=envelope.content,
                channel="gmail",
                channel_message_id=envelope.message_id,
            )
            result: TicketResult = (await support_app.process_batch([payload]))[0]
            if result.status == "rejected":
                await asyncio.to_thread(
                    gateway.label_message,
                    envelope.message_id,
                    gmail_settings.gmail_rejected_label,
                    True,
                )
                print(f"Ticket {result.ticket_id} rejected safely; no email was sent.")
                continue
            if result.status != "completed" or not result.human_approved or not result.final_reply:
                print(f"Ticket {result.ticket_id} was not approved; message remains unread and unsent.")
                continue
            if dry_run:
                print(f"DRY RUN: approved reply for ticket {result.ticket_id}; no email was sent.")
                continue
            confirmation = await asyncio.to_thread(
                input,
                f"Send the approved reply to {envelope.sender}? [y/N]: ",
            )
            if confirmation.strip().lower() not in {"y", "yes"}:
                print("Send cancelled; message remains unread.")
                continue
            await asyncio.to_thread(gateway.send_reply, envelope, result.final_reply)
            await asyncio.to_thread(
                gateway.label_message,
                envelope.message_id,
                gmail_settings.gmail_processed_label,
                True,
            )
            sent += 1
            print(f"Reply sent for ticket {result.ticket_id}; original message labeled as processed.")
        except (HttpError, OSError, ValueError, PermissionError) as exc:
            logging.getLogger(__name__).error(
                "Gmail message %s failed safely (%s)", message_id, type(exc).__name__
            )
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Process test-prefixed Gmail messages once.")
    parser.add_argument(
        "--send",
        action="store_true",
        help="Allow sending after application approval and a second explicit confirmation.",
    )
    args = parser.parse_args()
    count = asyncio.run(process_inbox(dry_run=not args.send))
    print(f"Gmail workflow complete. Replies sent: {count}")


if __name__ == "__main__":
    main()
