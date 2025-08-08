from datetime import datetime, timezone

def minutes_since(ts: datetime) -> float:
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 60.0
