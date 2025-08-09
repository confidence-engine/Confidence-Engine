from telegram_bot import format_alpha_message


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


