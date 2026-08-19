# Security Policy

## Project scope

This repository is an educational enterprise reference implementation. It is not certified for PHI,
not a medical device, and not a substitute for legal, compliance, coding, clinical, or reimbursement
review.

## Supported version

Security fixes are applied to the latest commit on `main` while the project is actively maintained.

## Reporting a vulnerability

Do not open a public issue containing a secret, credential, real patient information, exploit details,
or sensitive trace. Use GitHub's private vulnerability-reporting feature if it is enabled for the
repository. Otherwise, contact the repository owner privately through an approved channel and include
only synthetic reproduction data.

Please report:

- the affected commit and component;
- the security impact;
- minimal reproduction steps using synthetic data;
- whether credentials or PHI may have been exposed; and
- any suggested mitigation.

Revoke exposed credentials immediately and follow the relevant incident-response process. Removing a
secret from the latest commit does not remove it from Git history.

## Data-handling rules

- Never commit `.env`, service-account JSON, logs, traces, runtime output, or real support emails.
- Use synthetic data in development, tests, screenshots, demonstrations, and issue reports.
- Treat model providers, web search, tracing, and spreadsheets as external data recipients.
- Keep raw-to-mask mappings ticket-local and ephemeral.
- Do not log prompts, replies, email text, PHI tokens, or re-identification mappings in audit sinks.
- Review source licenses and redistribution rights before changing repository visibility.

## Production requirements

Before processing regulated data, perform a formal threat model and HIPAA risk analysis; use
BAA-eligible vendors; implement KMS/HSM-backed secrets, encryption, tenant isolation, least privilege,
private networking and egress controls, immutable auditing, retention/deletion policies, incident
response, content governance, continuous evaluation, and compliance approval.

The safeguards in this repository reduce risk but do not establish compliance by themselves.
