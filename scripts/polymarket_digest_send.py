#!/usr/bin/env python3
"""
Polymarket-only digest runner that fetches via the bridge (PPLX-backed),
sends to Telegram and Discord, writes a Markdown file, and auto-commits
and pushes the Markdown by default.

Env:
- TB_ENABLE_POLYMARKET=1 to enable fetching (default: enabled if unset)
- TB_NO_TELEGRAM=1 to skip Telegram send
- TB_NO_DISCORD=1 to skip Discord send
- TB_AUTOCOMMIT_DOCS=0 to disable git add/commit/push of the markdown file (default: enabled)
- TB_POLYMARKET_* filters and display toggles are respected by the bridge/formatters
- PPLX_API_KEY required for PPLX provider used by the bridge

Usage:
  # Safe dry run (no sends, writes markdown only)
  TB_NO_TELEGRAM=1 TB_NO_DISCORD=1 TB_ENABLE_POLYMARKET=1 \
  python3 scripts/polymarket_digest_send.py --debug

  # Live (requires TELEGRAM/DISCORD env configured)
  TB_ENABLE_POLYMARKET=1 \
  python3 scripts/polymarket_digest_send.py
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lightweight .env loader
def _load_dotenv_if_present():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            key, val = s.split('=', 1)
            key = key.strip(); val = val.strip().strip('"').strip("'")
            if key and (key not in os.environ or os.environ.get(key, '') == ''):
                os.environ[key] = val
    except Exception:
        pass

_load_dotenv_if_present()

# Imports that rely on project path
from scripts.polymarket_bridge import discover_from_env as discover_polymarket
from scripts import tg_digest_formatter as tg_fmt
from scripts.discord_formatter import digest_to_discord_embeds
from scripts.tg_sender import send_telegram_text
from scripts.discord_sender import send_discord_digest, send_discord_digest_to


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def _render_md(polymarket: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append('# Polymarket — Standalone Digest\n')
    if not polymarket:
        lines.append('_No qualifying markets found._')
        return "\n".join(lines)
    for pm in polymarket:
        title = pm.get('title') or 'Crypto market'
        stance = pm.get('stance') or 'Stand Aside'
        readiness = pm.get('readiness') or 'Later'
        edge = pm.get('edge_label') or 'in-line'
        # Title line (plain, no bullets/bold)
        lines.append(f"{title}")
        # Header line with stance/timing/edge
        lines.append(f"{stance} | {readiness} | {edge}")
        # Action (prefer implied side if present)
        implied = str(pm.get('implied_side') or '').strip().upper()
        if implied == 'YES':
            action = 'Buy YES'
        elif implied == 'NO':
            action = 'Buy NO'
        else:
            action = 'Take the bet' if str(stance).lower() == 'engage' else 'Stand Aside'
        lines.append(f"Action: {action}")
        # (Why removed as per request)
        # Outcome: always include; probability remains env-gated
        out_label = pm.get('outcome_label') or pm.get('implied_side') or '-'
        if os.getenv('TB_POLYMARKET_NUMBERS_IN_CHAT', '0') == '1' and os.getenv('TB_POLYMARKET_SHOW_PROB', '0') == '1':
            try:
                pct = pm.get('implied_pct')
                if isinstance(pct, int):
                    out_label = f"{out_label} ({pct}%)"
                else:
                    imp = pm.get('implied_prob')
                    if isinstance(imp, (int, float)):
                        out_label = f"{out_label} ({float(imp)*100:.0f}%)"
            except Exception:
                pass
        lines.append(f"Outcome: {out_label}")
    return "\n".join(lines)


def _git_autocommit_push(file_path: Path, debug: bool = False) -> None:
    try:
        # Only operate if inside a git repo
        if not (ROOT / '.git').exists():
            return
        # Stage file
        subprocess.run(['git', 'add', str(file_path)], cwd=str(ROOT), check=False)
        # Commit
        msg = f"docs(polymarket): update digest {datetime.utcnow().isoformat(timespec='seconds')}Z"
        subprocess.run(['git', 'commit', '-m', msg], cwd=str(ROOT), check=False)
        # Push
        subprocess.run(['git', 'push'], cwd=str(ROOT), check=False)
    except Exception as e:
        if debug:
            print(f"[autocommit] skipped or failed: {e}")


def main() -> int:
    debug = os.getenv('TB_POLYMARKET_DEBUG', '0') == '1' or '--debug' in sys.argv
    # Force-enable polymarket for this runner unless explicitly disabled
    if 'TB_ENABLE_POLYMARKET' not in os.environ:
        os.environ['TB_ENABLE_POLYMARKET'] = '1'
    # Use only the plain PPLX_API_KEY (dedicated) for this runner unless explicitly disabled
    if 'TB_POLYMARKET_PPLX_USE_PLAIN_ONLY' not in os.environ:
        os.environ['TB_POLYMARKET_PPLX_USE_PLAIN_ONLY'] = '1'
    enable_poly = os.getenv('TB_ENABLE_POLYMARKET', '1') == '1'
    if not enable_poly:
        if debug:
            print('[polymarket] disabled by TB_ENABLE_POLYMARKET=0')
        return 0

    # 1) Fetch via bridge (PPLX-backed, env-controlled)
    try:
        polymarket_items = discover_polymarket()
    except Exception as e:
        if debug:
            print(f"[polymarket] fetch failed: {e}")
        polymarket_items = []

    ts = _now_iso()

    # 2) Telegram text using existing formatter (polymarket-only)
    try:
        tg_text = tg_fmt.render_digest(
            timestamp_utc=ts,
            weekly={},
            engine={},
            assets_ordered=[],
            assets_data={},
            options={
                'include_weekly': False,
                'include_engine': False,
                'provenance': {},
                'include_prices': False,
            },
            polymarket=polymarket_items,
        )
    except Exception as e:
        tg_text = "Polymarket — Standalone Digest\n\n_No qualifying markets found._" if not polymarket_items else f"Polymarket digest generated with {len(polymarket_items)} items."
        if debug:
            print(f"[tg] render failed: {e}")

    # 3) Discord embeds (polymarket-only)
    try:
        digest_data = {
            'timestamp': ts,
            'executive_take': None,
            'weekly': {},
            'engine': {},
            'assets': [],
            'polymarket': polymarket_items,
            'provenance': {},
        }
        embeds = digest_to_discord_embeds(digest_data)
    except Exception as e:
        embeds = []
        if debug:
            print(f"[discord] render failed: {e}")

    # 4) Write markdown file
    md = _render_md(polymarket_items)
    md_path = ROOT / 'polymarket_digest.md'
    try:
        md_path.write_text(md + ("\n" if not md.endswith("\n") else ""), encoding='utf-8')
        if debug:
            print(f"[file] wrote {md_path}")
    except Exception as e:
        if debug:
            print(f"[file] write failed: {e}")

    # 5) Send to Telegram/Discord unless suppressed
    if os.getenv('TB_NO_TELEGRAM', '0') != '1':
        try:
            send_telegram_text(tg_text)
            if debug:
                print('[tg] sent')
        except Exception as e:
            if debug:
                print(f"[tg] send failed: {e}")
    else:
        if debug:
            print('[tg] suppressed by TB_NO_TELEGRAM=1')

    if os.getenv('TB_NO_DISCORD', '0') != '1':
        try:
            if embeds:
                # Prefer dedicated polymarket webhook when provided
                poly_webhook = os.getenv('DISCORD_POLYMARKET_WEBHOOK_URL')
                if poly_webhook:
                    send_discord_digest_to(poly_webhook, embeds)
                else:
                    send_discord_digest(embeds)
                if debug:
                    print('[discord] sent')
        except Exception as e:
            if debug:
                print(f"[discord] send failed: {e}")
    else:
        if debug:
            print('[discord] suppressed by TB_NO_DISCORD=1')

    # 6) Optional auto-commit/push of markdown
    # Default-on: auto-commit unless explicitly disabled
    if os.getenv('TB_AUTOCOMMIT_DOCS', '1') != '0':
        _git_autocommit_push(md_path, debug=debug)
    else:
        if debug:
            print('[autocommit] disabled (set TB_AUTOCOMMIT_DOCS!=0 to enable)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
