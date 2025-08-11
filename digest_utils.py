"""
Digest utilities for multi-asset universe summaries.
"""

from typing import List, Dict, Optional


def format_universe_digest(payloads: List[Dict], top_n: int = 5) -> str:
    """
    Format a universe digest for Telegram.
    
    Args:
        payloads: List of payload dictionaries from symbol analysis
        top_n: Number of top results to include
        
    Returns:
        Formatted digest string
    """
    if not payloads:
        return "No signals found in universe scan."
    
    # Sort by absolute divergence (primary) and confidence (secondary)
    sorted_payloads = sorted(
        payloads,
        key=lambda p: (
            abs(p.get("timescale_scores", {}).get("combined_divergence", p.get("divergence", 0))),
            p.get("confidence", 0)
        ),
        reverse=True
    )
    
    # Take top N
    top_payloads = sorted_payloads[:top_n]
    
    # Build digest
    lines = [f"🔍 Universe Scan - Top {len(top_payloads)} Signals"]
    lines.append("")  # Empty line for spacing
    
    for i, payload in enumerate(top_payloads, 1):
        symbol = payload.get("symbol", "UNKNOWN")
        
        # Get divergence (prefer timescale combined, fallback to regular)
        divergence = payload.get("timescale_scores", {}).get("combined_divergence", payload.get("divergence", 0))
        confidence = payload.get("confidence", 0)
        
        # Get volume info
        volume_z = payload.get("volume_z", 0)
        if "timescale_scores" in payload:
            # Use short-term volume z-score
            volume_z = payload["timescale_scores"].get("short", {}).get("volume_z", volume_z)
        
        # Format divergence and confidence
        div_str = f"{divergence:+.2f}"
        conf_str = f"{confidence:.2f}"
        vol_str = f"{volume_z:+.1f}"
        
        # Build tags
        tags = []
        
        # Diversity adjustment
        diversity = payload.get("source_diversity", {})
        if diversity and abs(diversity.get("adjustment", 0)) >= 0.01:
            adj = diversity.get("adjustment", 0)
            tags.append(f"div {adj:+.2f}")
        
        # Cascade tag
        cascade = payload.get("cascade_detector", {})
        if cascade.get("tag") == "HYPE_ONLY":
            tags.append("hype -")
        
        # Timescale alignment
        timescales = payload.get("timescale_scores", {})
        if timescales:
            aligned = timescales.get("aligned_horizons", 0)
            total = 3  # short, mid, long
            tags.append(f"ts align {aligned}/{total}")
        
        # Position sizing
        sizing = payload.get("position_sizing", {})
        if sizing and sizing.get("target_R", 0) > 0:
            target_r = sizing.get("target_R", 0)
            tags.append(f"R{target_r:.2f}")
        
        # Build the line
        line = f"{i}. {symbol} — gap {div_str}, conf {conf_str}, VolZ {vol_str}"
        
        if tags:
            line += f"; tags: {', '.join(tags)}"
        
        lines.append(line)
    
    # Add summary stats
    total_signals = len(payloads)
    positive_div = sum(1 for p in payloads if p.get("divergence", 0) > 0)
    negative_div = sum(1 for p in payloads if p.get("divergence", 0) < 0)
    
    lines.append("")
    lines.append(f"📊 Summary: {total_signals} total, {positive_div} bullish, {negative_div} bearish")
    
    return "\n".join(lines)


def format_compact_digest(payloads: List[Dict], top_n: int = 5) -> str:
    """
    Format a more compact universe digest.
    
    Args:
        payloads: List of payload dictionaries
        top_n: Number of top results to include
        
    Returns:
        Compact digest string
    """
    if not payloads:
        return "No signals found."
    
    # Sort by absolute divergence
    sorted_payloads = sorted(
        payloads,
        key=lambda p: abs(p.get("divergence", 0)),
        reverse=True
    )
    
    top_payloads = sorted_payloads[:top_n]
    
    lines = [f"Top {len(top_payloads)} signals:"]
    
    for payload in top_payloads:
        symbol = payload.get("symbol", "UNKNOWN")
        divergence = payload.get("divergence", 0)
        confidence = payload.get("confidence", 0)
        
        line = f"{symbol}: {divergence:+.2f} (conf {confidence:.2f})"
        lines.append(line)
    
    return "\n".join(lines)


def validate_digest_length(digest: str, max_length: int = 4000) -> bool:
    """
    Validate that digest fits within Telegram message limits.
    
    Args:
        digest: Digest string
        max_length: Maximum allowed length
        
    Returns:
        True if digest fits, False otherwise
    """
    return len(digest) <= max_length


def truncate_digest(digest: str, max_length: int = 4000) -> str:
    """
    Truncate digest to fit within limits.
    
    Args:
        digest: Digest string
        max_length: Maximum allowed length
        
    Returns:
        Truncated digest string
    """
    if len(digest) <= max_length:
        return digest
    
    # Try to truncate at line boundaries
    lines = digest.split('\n')
    truncated_lines = []
    current_length = 0
    
    for line in lines:
        if current_length + len(line) + 1 <= max_length - 20:  # Leave room for truncation notice
            truncated_lines.append(line)
            current_length += len(line) + 1
        else:
            break
    
    if truncated_lines:
        truncated_lines.append("... (truncated)")
        return '\n'.join(truncated_lines)
    else:
        # If even first line is too long, truncate it
        return digest[:max_length-3] + "..."
