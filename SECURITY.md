# Security Policy

We value responsible disclosure. Please do not file public issues for security-sensitive reports.

## Reporting a vulnerability
- Email: open a private report via GitHub Security Advisories or email the maintainer (repo owner) with subject: "CONFIDENCE-ENGINE SECURITY".
- Include: affected files, reproduction steps, impact assessment, and any logs. Do not include secrets.
- We aim to acknowledge within 72 hours and provide an initial assessment within 7 days.

## Scope and guidelines
- Secrets: API keys, tokens, chat/webhook URLs, or credentials found in code or history.
- Supply chain: dependency vulnerabilities, malicious packages, or lockfile poisoning.
- CI/automation: unintended external sends or git pushes without explicit env gating.
- Data privacy: accidental PII in logs or artifacts.

Out of scope (unless combined with another issue):
- Rate limiting and generic DoS without a concrete bypass.
- Social engineering of maintainers or providers.

## Operational safety defaults
- Safe-by-default flags are documented in `.env.example` (no Telegram/Discord sends, no trading). Please report any bypass.
- Auto-commit only targets non-code artifacts and is env-gated.
