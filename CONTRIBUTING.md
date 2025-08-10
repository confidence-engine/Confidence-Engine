# Contributing

- Python 3.11; keep secrets in .env (never commit).
- Run before PR:
```
flake8 .
black --check .
python3 -m pytest -q
```
- Respect config precedence; don’t remove payload keys.
- For new features, add pure functions + tests.
- Keep Telegram messages <4000 chars; plain text default.
