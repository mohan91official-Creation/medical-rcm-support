from email.message import EmailMessage

from gmail_workflow import extract_message_body, is_automated_message


def test_plain_text_body_is_extracted() -> None:
    message = EmailMessage()
    message.set_content("Synthetic CO-16 test")
    assert extract_message_body(message) == "Synthetic CO-16 test"


def test_attachment_text_is_not_included() -> None:
    message = EmailMessage()
    message.set_content("Safe body")
    message.add_attachment(b"secret", maintype="text", subtype="plain", filename="secret.txt")
    assert extract_message_body(message) == "Safe body"


def test_auto_submitted_message_is_rejected() -> None:
    message = EmailMessage()
    message["From"] = "robot@example.com"
    message["Auto-Submitted"] = "auto-replied"
    assert is_automated_message(message)


def test_human_message_is_allowed() -> None:
    message = EmailMessage()
    message["From"] = "customer@example.com"
    assert not is_automated_message(message)
