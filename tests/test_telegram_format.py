from telegram_bot import format_alpha_message, send_message
import types
import sys
import json


def test_format_alpha_message_contains_core_fields():
    payload = {
        "symbol": "BTC/USD",
        "confidence": 0.75,
        "divergence": 1.23,
        "divergence_threshold": 1.0,
        "volume_z": 0.5,
        "action": "BUY",
        "alpha_summary": "Strong positive narrative vs. lagging price.",
        "alpha_next_steps": "Buy small, add on strength.",
        "btc_filtered_headlines": "[]",
        "relevance_details": "{\"accepted\": []}",
    }
    msg = format_alpha_message(payload)

    assert "Tracer Bullet" in msg
    assert "BTC/USD" in msg
    assert "Action" in msg
    assert "Conf" in msg
    assert "Gap" in msg


def test_send_message_truncates_and_handles_errors(monkeypatch):
    # Force TB_NO_TELEGRAM off to exercise HTTP code path
    monkeypatch.delenv("TB_NO_TELEGRAM", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    # Ensure default parse mode is plain text
    monkeypatch.delenv("TELEGRAM_PARSE_MODE", raising=False)

    # Stub requests.post to simulate 429 and 400
    import telegram_bot as tb

    class Resp:
        def __init__(self, status_code, text, headers=None):
            self.status_code = status_code
            self.text = text
            self.headers = headers or {}

        def json(self):
            try:
                return json.loads(self.text)
            except Exception:
                return {"description": self.text[:100]}

    calls = {"payloads": []}

    def fake_post(url, json=None, timeout=10):
        calls["payloads"].append(json)
        # First call: 429
        if len(calls["payloads"]) == 1:
            return Resp(429, "{\"description\": \"Too Many Requests\"}", {"Retry-After": "2"})
        # Second call: 400
        return Resp(400, "{\"description\": \"Bad Request\"}")

    monkeypatch.setattr(tb.requests, "post", fake_post)

    long_text = "x" * 5000
    ok1 = send_message(long_text)
    ok2 = send_message("short")

    assert ok1 is False and ok2 is False
    assert len(calls["payloads"][0]["text"]) == 4000


