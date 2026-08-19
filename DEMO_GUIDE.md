# Buildathon Demo Guide

This is a repeatable five-minute demonstration of the project's strongest engineering controls. Use
synthetic data only. Keep `.env`, service-account credentials, provider dashboards, and raw traces out
of screen recordings.

## Before the presentation

1. Activate `.venv` and confirm the terminal shows `(.venv)`.
2. Set `AUTO_APPROVE=false` so the human approval gate is visible.
3. Keep the `tickets` worksheet and the LangSmith project open in separate browser tabs.
4. Run the quality checks once before presenting:

   ```powershell
   python -m ruff check .
   python -m pytest -q --basetemp .venv\pytest-temp
   python evaluate_retrieval.py --tune
   ```

5. Never display `.env` or `service-account-google.json`.

## Five-minute judge script

### 0:00–0:40 — Problem and differentiator

“Medical-billing support contains sensitive data and payer-specific rules. This system treats every
email as hostile, masks PHI locally, retrieves governed evidence, creates an isolated CrewAI workflow,
and requires human approval before release.”

Show the architecture diagram in `README.md` and point out that raw PHI does not go to the LLM, search,
tracing, or audit sheet.

### 0:40–1:30 — Security boundary

Run the included demo:

```powershell
python app.py
```

Highlight two concurrent tickets:

- The legitimate CO-16 request passes the domain gate and is masked.
- The malicious instruction is rejected before retrieval or any LLM call.

Point out the ticket-local masking vault and fail-closed handling of unknown mask tokens.

### 1:30–2:40 — Evidence-first answer

In the valid ticket output, highlight:

- `source: hybrid`
- the confidence score
- citations to the internal RCM reference and CARC source
- no unrelated LCD source for the CO-16 question

Explain: “Retrieval combines local spaCy embeddings, FAISS, BM25, reciprocal-rank fusion, exact
CARC/RARC matching, source routing, and focus reranking. Serper runs only when local confidence is below
0.70, using a masked query.”

### 2:40–3:30 — Multi-agent workflow and HITL

At the approval prompt, review the proposed reply and enter `y`.

Explain that each ticket receives fresh triage, response, and QA agents with Pydantic-validated outputs.
OpenAI is primary and NVIDIA is the fallback. The approval prompt is protected by an async lock so
concurrent tickets cannot overlap human decisions.

### 3:30–4:15 — Privacy-safe observability

Show:

- LangSmith: masked values such as `<PERSON_1>`, never the synthetic raw identity.
- Google Sheets: lifecycle stages, retrieval confidence, tokens, cost, duration, and error class—without
  email content or the re-identification map.

State clearly that these integrations are demonstration sinks, not a HIPAA system of record.

### 4:15–5:00 — Measured quality and roadmap

Show the green GitHub Actions Quality Gate and the retrieval benchmark result:

- 10 automated security/retrieval tests
- objective score approximately 0.9556 on the included golden set
- every evaluation case finds relevant evidence
- zero forbidden-source contamination in that set

Close with the production roadmap: BAA-eligible services, KMS/HSM secrets, durable orchestration,
tenant isolation, immutable auditing, authoritative content governance, and continuous red-team/eval
datasets.

## Judge questions — short answers

**Is this HIPAA compliant?**  
No codebase alone is HIPAA compliant. This reference demonstrates PHI minimization and security
boundaries; production requires BAAs, risk analysis, policies, access controls, infrastructure, and
compliance approval.

**Why CrewAI?**  
It makes responsibilities explicit: triage, evidence-grounded response, and independent QA. The system
does not delegate deterministic security decisions to agents.

**How do you prevent cross-ticket PHI leakage?**  
Every ticket receives a new state object, masking vault, Crew, agents, tasks, and usage ledger. Final
formatting can resolve only tokens stored in that ticket's vault.

**What happens if OpenAI fails?**  
The LangChain runnable attempts ChatNVIDIA through `with_fallbacks`. A ticket-level exception is isolated
and returned as a safe failure rather than terminating the batch.

**Why not trust the LLM judge alone?**  
The judge supplements deterministic checks. Guardrails, masking, token validation, retrieval routing,
audit allowlisting, and human release approval remain outside model judgment.

## Presentation safety checklist

- Use synthetic names, identifiers, and scenarios only.
- Hide all API keys and credential files.
- Do not claim legal, clinical, coding, reimbursement, or HIPAA certification.
- Do not claim the small included evaluation set proves production accuracy.
- Confirm payer, jurisdiction, effective date, and authoritative source before real operational use.
- Review redistribution rights before making bundled reference documents public.
