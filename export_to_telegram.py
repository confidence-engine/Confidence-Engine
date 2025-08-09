import json, os
from glob import glob
from telegram_bot import send_message

def latest_run_json_path() -> str:
    files = sorted(glob("runs/*.json"), key=os.path.getmtime, reverse=True)
    return files[0] if files else ""

def send_latest_run_to_telegram(path: str = "") -> bool:
    fp = path or latest_run_json_path()
    if not fp:
        print("[ExportTelegram] No runs/*.json found"); return False
    with open(fp, "r", encoding="utf-8") as f:
        p = json.load(f)
    msg = (
        f"Tracer Bullet • {p.get('symbol','BTC/USD')}\n"
        f"Action: {p.get('action','HOLD')} | Gap: {p.get('divergence',0):+0.2f} "
        f"(trigger {p.get('divergence_threshold',1.0):.2f}) | "
        f"Conf: {p.get('confidence',0):.2f} | VolZ: {p.get('volume_z',0):+0.2f}\n\n"
        f"{p.get('alpha_summary','')}\n\n"
        f"{p.get('alpha_next_steps','')}"
    ).strip()
    return send_message(msg)

if __name__ == "__main__":
    print("[ExportTelegram] sent:", send_latest_run_to_telegram())
