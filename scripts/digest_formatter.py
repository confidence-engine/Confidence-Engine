#!/usr/bin/env python3
"""
Human-readable digest formatter for Tracer Bullet.

Converts technical analysis results into a conversational, number-free, 
crypto-first digest with entries/exits/levels-to-watch and risk-based sizing.
"""

from typing import Dict, List, Optional


def render_digest(summary: dict) -> str:
    """
    Convert the run summary into a human-readable digest.
    
    Args:
        summary: Dictionary containing run summary with payloads and metadata
        
    Returns:
        Formatted human-readable digest string
    """
    payloads = summary.get("payloads", [])
    if not payloads:
        return "No signals found in universe scan."
    
    # Separate crypto and non-crypto assets
    crypto_payloads = [p for p in payloads if p.get("symbol_type") == "crypto"]
    other_payloads = [p for p in payloads if p.get("symbol_type") != "crypto"]
    
    # Find BTC and ETH payloads
    btc_payload = next((p for p in crypto_payloads if p.get("symbol", "").startswith("BTC")), None)
    eth_payload = next((p for p in crypto_payloads if p.get("symbol", "").startswith("ETH")), None)
    
    # Other crypto assets
    other_crypto = [p for p in crypto_payloads if p not in [btc_payload, eth_payload] if p]
    
    # Build the digest
    lines = []
    
    # Header
    timestamp = summary.get("timestamp_iso", "")
    version = summary.get("version", "")
    header = f"Tracer Bullet {version} • {timestamp}"
    lines.append(header)
    lines.append("")
    
    # Executive take
    lines.append("## Executive Take")
    
    # Determine overall market sentiment
    bullish_count = sum(1 for p in payloads if p.get("divergence", 0) > 0)
    bearish_count = sum(1 for p in payloads if p.get("divergence", 0) < 0)
    
    if bullish_count > bearish_count:
        market_sentiment = "broadly positive"
    elif bearish_count > bullish_count:
        market_sentiment = "broadly cautious"
    else:
        market_sentiment = "mixed"
    
    # Add crypto-first executive summary
    exec_summary = f"Market sentiment is {market_sentiment} today."
    
    # Add crypto-specific context if available
    if btc_payload or eth_payload:
        exec_summary += " Crypto assets are"
        
        if btc_payload and btc_payload.get("divergence", 0) > 0:
            exec_summary += " showing strength in Bitcoin"
        elif btc_payload:
            exec_summary += " showing weakness in Bitcoin"
            
        if eth_payload:
            if btc_payload:
                exec_summary += " while"
                
            if eth_payload.get("divergence", 0) > 0:
                exec_summary += " Ethereum displays positive momentum"
            else:
                exec_summary += " Ethereum shows consolidation patterns"
    
    lines.append(exec_summary)
    lines.append("")
    
    # BTC/USD section
    if btc_payload:
        _add_asset_section(lines, btc_payload)
    
    # ETH/USD section
    if eth_payload:
        _add_asset_section(lines, eth_payload)
    
    # Other majors (if present)
    if other_crypto:
        lines.append("## Other Crypto Assets")
        for payload in other_crypto:
            symbol = payload.get("symbol", "Unknown")
            sentiment = _get_sentiment_description(payload)
            confidence = _get_confidence_description(payload)
            lines.append(f"**{symbol}**: {sentiment} with {confidence} confidence.")
        lines.append("")
    
    # Equities (background)
    if other_payloads:
        lines.append("## Equities Background")
        equity_sentiment = _get_equity_sentiment(other_payloads)
        lines.append(equity_sentiment)
        lines.append("")
    
    # Playbook
    lines.append("## Playbook")
    lines.append("* Focus on high-confidence setups with clear invalidation levels")
    lines.append("* Size positions according to conviction and market volatility")
    lines.append("* Be patient with entries and exits; avoid chasing momentum")
    
    return "\n".join(lines)


def _add_asset_section(lines: List[str], payload: Dict) -> None:
    """
    Add a detailed asset section to the digest.
    
    Args:
        lines: List of digest lines to append to
        payload: Asset payload dictionary
    """
    symbol = payload.get("symbol", "Unknown")
    
    # Get descriptive values
    sentiment = _get_sentiment_description(payload)
    confidence = _get_confidence_description(payload)
    volume = _get_volume_description(payload)
    alignment = _get_alignment_description(payload)
    quality = _get_quality_description(payload)
    readiness = _get_readiness_description(payload)
    sizing = _get_sizing_description(payload)
    
    # Add section header
    lines.append(f"## {symbol} — {sizing} Risk | {readiness} | {_get_action_description(payload)}")
    
    # What's going on
    lines.append("### What's Going On")
    narrative = f"Price action is {sentiment} with {confidence} confidence. "
    narrative += f"Market participation is {volume}. "
    
    # Add rationale based on available data
    if payload.get("source_diversity", {}).get("unique", 0) > 2:
        narrative += "Multiple sources confirm this view."
    elif payload.get("cascade_detector", {}).get("tag") == "HYPE_ONLY":
        narrative += "Narrative appears to be driven primarily by market hype."
    else:
        narrative += "The current setup shows a clear directional bias."
    
    lines.append(narrative)
    lines.append("")
    
    # Why we see it
    lines.append("### Why We See It")
    lines.append(f"* Cross-horizon analysis: {alignment}")
    lines.append(f"* Participation: {volume}")
    lines.append(f"* Signal quality: {quality}")
    
    # Price vs story
    price_vs_narrative = next((c for c in payload.get("confirmation_checks", []) 
                             if c.get("name") == "price_vs_narrative"), {})
    if price_vs_narrative.get("passed", False):
        price_story = "Price action confirms the narrative"
    else:
        price_story = "Price action and narrative show some disconnect"
    lines.append(f"* Price vs story: {price_story}")
    lines.append("")
    
    # Levels to watch
    lines.append("### Levels to Watch")
    if payload.get("divergence", 0) > 0:
        lines.append("* Watch for continuation above recent highs")
        lines.append("* Key support at the recent consolidation base")
    else:
        lines.append("* Watch for support at recent lows")
        lines.append("* Key resistance at the recent distribution zone")
    lines.append("")
    
    # Entry ideas
    lines.append("### Entry Ideas")
    if payload.get("divergence", 0) > 0:
        lines.append("* Buy on pullbacks to support levels with tight stops")
        lines.append("* Scale in on confirmation of trend continuation")
    else:
        lines.append("* Sell on rallies to resistance with defined risk")
        lines.append("* Look for breakdown confirmation before full position")
    lines.append("")
    
    # Exit and invalidation
    lines.append("### Exit and Invalidation")
    if payload.get("divergence", 0) > 0:
        lines.append("Take profit at major resistance levels or when momentum wanes. Invalidation occurs if price breaks below key support with volume.")
    else:
        lines.append("Take profit at major support levels or when downward momentum exhausts. Invalidation occurs if price breaks above key resistance with strong volume.")
    lines.append("")
    
    # Sizing
    lines.append("### Sizing")
    lines.append(f"Recommended exposure: {sizing} based on current risk/reward profile.")
    lines.append("")


def _get_sentiment_description(payload: Dict) -> str:
    """
    Get a descriptive sentiment label based on divergence.
    """
    divergence = payload.get("divergence", 0)
    
    if divergence > 0.8:
        return "strongly bullish"
    elif divergence > 0.3:
        return "bullish"
    elif divergence > 0:
        return "mildly bullish"
    elif divergence > -0.3:
        return "mildly bearish"
    elif divergence > -0.8:
        return "bearish"
    else:
        return "strongly bearish"


def _get_confidence_description(payload: Dict) -> str:
    """
    Get a descriptive confidence label.
    """
    confidence = payload.get("confidence", 0)
    
    if confidence > 0.85:
        return "very high"
    elif confidence > 0.75:
        return "strong"
    elif confidence > 0.65:
        return "moderate"
    else:
        return "low"


def _get_volume_description(payload: Dict) -> str:
    """
    Get a descriptive volume label.
    """
    volume_z = payload.get("volume_z", 0)
    
    if volume_z > 0.5:
        return "hot"
    elif volume_z > -0.5:
        return "normal"
    else:
        return "quiet"


def _get_alignment_description(payload: Dict) -> str:
    """
    Get a descriptive alignment label based on timescale alignment.
    """
    aligned = payload.get("timescale_scores", {}).get("aligned_horizons", 0)
    total = 3  # short, mid, long
    
    if aligned == total:
        return "strong alignment across all timeframes"
    elif aligned == 2:
        return "mixed alignment with some timeframe divergence"
    else:
        return "fragmented across different timeframes"


def _get_quality_description(payload: Dict) -> str:
    """
    Get a descriptive quality label based on source diversity and cascade detection.
    """
    diversity = payload.get("source_diversity", {})
    cascade = payload.get("cascade_detector", {})
    
    if diversity.get("unique", 0) > 2 and cascade.get("tag", "") != "HYPE_ONLY":
        return "broad-based"
    elif diversity.get("unique", 0) > 1:
        return "mixed sources"
    elif cascade.get("tag") == "HYPE_ONLY":
        return "hype-prone"
    else:
        return "limited confirmation"


def _get_readiness_description(payload: Dict) -> str:
    """
    Get a readiness label based on confidence and confirmation checks.
    """
    confidence = payload.get("confidence", 0)
    checks_passed = sum(1 for c in payload.get("confirmation_checks", []) if c.get("passed", False))
    
    if confidence > 0.75 and checks_passed >= 2:
        return "Now"
    elif confidence > 0.65 and checks_passed >= 1:
        return "Near"
    else:
        return "Later"


def _get_sizing_description(payload: Dict) -> str:
    """
    Get a sizing band description based on position sizing.
    """
    target_r = payload.get("position_sizing", {}).get("target_R", 0)
    
    if target_r > 0.6:
        return "High"
    elif target_r > 0.3:
        return "Medium"
    else:
        return "Low"


def _get_action_description(payload: Dict) -> str:
    """
    Get an action description based on divergence and confidence.
    """
    divergence = payload.get("divergence", 0)
    confidence = payload.get("confidence", 0)
    
    if divergence > 0 and confidence > 0.7:
        return "Buy"
    elif divergence < 0 and confidence > 0.7:
        return "Sell"
    elif divergence > 0:
        return "Watch for Buy"
    elif divergence < 0:
        return "Watch for Sell"
    else:
        return "Monitor"


def _get_equity_sentiment(equity_payloads: List[Dict]) -> str:
    """
    Get a summary of equity market sentiment.
    """
    if not equity_payloads:
        return "No equity data available."
    
    bullish = sum(1 for p in equity_payloads if p.get("divergence", 0) > 0)
    bearish = sum(1 for p in equity_payloads if p.get("divergence", 0) < 0)
    total = len(equity_payloads)
    
    if bullish > bearish and bullish > total / 2:
        return "Equity markets showing broad strength with positive sentiment."
    elif bearish > bullish and bearish > total / 2:
        return "Equity markets under pressure with cautious sentiment prevailing."
    else:
        return "Equity markets mixed with no clear directional bias."