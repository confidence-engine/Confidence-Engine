from db import list_runs, get_run
import json
import sys

def main():
    runs = list_runs(limit=20)
    print("Recent runs:")
    for r in runs:
        rid, ts, sym, act, rsn, summ = r
        print(f"- id={rid} [{ts}] {sym} → {act} ({rsn}) | {summ}")

    if len(sys.argv) > 1:
        rid = int(sys.argv[1])
        row = get_run(rid)
        if not row:
            print(f"\nRun {rid} not found.")
            return
        print(f"\nFull run {rid}:")
        for k, v in row.items():
            if k in ("raw_headlines", "btc_filtered_headlines"):
                try:
                    v = json.dumps(json.loads(v), indent=2, ensure_ascii=False)
                except Exception:
                    pass
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
