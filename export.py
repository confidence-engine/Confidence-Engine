import os
import json
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def export_run_json(run_id: int, payload: dict) -> str:
    ensure_dir("runs")
    out_path = os.path.join("runs", f"{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path

def save_bars_csv(run_id: int, bars_df: pd.DataFrame) -> str:
    ensure_dir("bars")
    out_path = os.path.join("bars", f"{run_id}.csv")
    df = bars_df.copy()
    df.index.name = "timestamp"
    df.to_csv(out_path)
    return out_path

def save_accepted_txt(run_id: int, accepted_with_src: list) -> str:
    """
    Persist accepted headline texts and sources for quick human review.
    One line per accepted headline:
    [source] relevance | headline

    Example line:
    [perplexity] 0.473 | Bitcoin forms higher low, technicals hint at possible rally toward all-time highs
    """
    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"{run_id}_accepted.txt")
    lines = []
    for item in accepted_with_src:
        src = item.get("source", "?")
        rel = item.get("relevance", 0.0)
        h = item.get("headline", "")
        lines.append(f"[{src}] {rel:.3f} | {h}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

