"""
Local sentiment analyzer - simplified implementation for hybrid agent
"""
import logging
from typing import List, Tuple, Dict, Any
from sentiment_utils import trimmed_mean, drop_outliers

logger = logging.getLogger(__name__)

def get_local_sentiment_analysis(headlines: List[str]) -> float:
    """
    Analyze sentiment of headlines locally.
    Returns sentiment score between -1 (negative) and 1 (positive).
    """
    if not headlines:
        return 0.0
    
    # Simple keyword-based sentiment for now
    scores = []
    positive_words = ['bull', 'bullish', 'up', 'gain', 'rise', 'surge', 'rally', 'positive', 'growth', 'strong']
    negative_words = ['bear', 'bearish', 'down', 'fall', 'drop', 'crash', 'negative', 'decline', 'weak', 'sell']
    
    for headline in headlines:
        headline_lower = headline.lower()
        score = 0.0
        
        for word in positive_words:
            if word in headline_lower:
                score += 0.1
        
        for word in negative_words:
            if word in headline_lower:
                score -= 0.1
        
        # Cap scores
        score = max(-1.0, min(1.0, score))
        scores.append(score)
    
    if not scores:
        return 0.0
    
    # Use robust aggregation
    clean_scores, _ = drop_outliers(scores)
    if clean_scores:
        return trimmed_mean(clean_scores)
    else:
        return 0.0

def get_local_narrative_summary(headlines: List[str]) -> str:
    """
    Generate a narrative summary from headlines.
    """
    if not headlines:
        return "No headlines available"
    
    # Simple concatenation for now
    return " | ".join(headlines[:3])

def sentiment_via_local(headlines: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Get sentiment via local analysis.
    Returns (sentiment_score, metadata)
    """
    sentiment = get_local_sentiment_analysis(headlines)
    metadata = {
        'method': 'local_keywords',
        'headlines_count': len(headlines),
        'confidence': min(1.0, len(headlines) / 10.0)  # Higher confidence with more headlines
    }
    
    logger.info(f"Local sentiment analysis: {sentiment:.3f} (from {len(headlines)} headlines)")
    
    return sentiment, metadata