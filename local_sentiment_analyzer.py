#!/usr/bin/env python3
"""
Local Sentiment Analyzer - Perplexity Replacement
Provides local sentiment analysis without external API dependencies
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
from finbert import sentiment_robust, sentiment_score

logger = logging.getLogger(__name__)

class LocalSentimentAnalyzer:
    """
    Local sentiment analysis using FinBERT only
    Replaces Perplexity API calls with local processing
    """
    
    def __init__(self):
        self.name = "LocalSentimentAnalyzer"
        logger.info(f"✅ {self.name} initialized - No external APIs required")
    
    def analyze_headlines_sentiment(self, headlines: List[str]) -> Tuple[float, Optional[str]]:
        """
        Analyze sentiment of headlines using local FinBERT model
        Returns (sentiment_score, error_message)
        
        Args:
            headlines: List of news headlines
            
        Returns:
            Tuple of (sentiment_score [0-1], error_message or None)
        """
        if not headlines:
            return 0.5, None  # Neutral default
            
        try:
            # Use robust FinBERT analysis with outlier removal
            finbert_score, kept_scores, dropped_scores = sentiment_robust(headlines)
            
            # Convert from [-1, 1] to [0, 1] range to match Perplexity format
            sentiment_normalized = (finbert_score + 1.0) / 2.0
            
            # Clamp to valid range
            sentiment_normalized = max(0.0, min(1.0, sentiment_normalized))
            
            logger.info(f"📊 Local sentiment analysis: {sentiment_normalized:.3f} "
                       f"(FinBERT: {finbert_score:.3f}, kept: {len(kept_scores)}, "
                       f"dropped: {len(dropped_scores)})")
            
            return sentiment_normalized, None
            
        except Exception as e:
            error_msg = f"Local sentiment analysis failed: {e}"
            logger.error(error_msg)
            return 0.5, error_msg  # Neutral fallback on error
    
    def get_narrative_summary(self, headlines: List[str]) -> Tuple[str, float]:
        """
        Generate a narrative summary from headlines using local analysis
        
        Args:
            headlines: List of news headlines
            
        Returns:
            Tuple of (narrative_text, confidence_score)
        """
        if not headlines:
            return "No news available", 0.0
            
        try:
            # Analyze sentiment
            sentiment_score_norm, _ = self.analyze_headlines_sentiment(headlines)
            
            # Generate simple narrative based on sentiment
            if sentiment_score_norm > 0.65:
                sentiment_desc = "strongly bullish"
                confidence = 0.75
            elif sentiment_score_norm > 0.55:
                sentiment_desc = "bullish"
                confidence = 0.65
            elif sentiment_score_norm < 0.35:
                sentiment_desc = "strongly bearish"
                confidence = 0.75
            elif sentiment_score_norm < 0.45:
                sentiment_desc = "bearish"
                confidence = 0.65
            else:
                sentiment_desc = "neutral"
                confidence = 0.5
            
            # Count keywords that might indicate strength
            combined_text = " ".join(headlines).lower()
            strength_keywords = [
                'surge', 'rally', 'breakout', 'bullish', 'gains', 'up', 'rise', 'positive',
                'drop', 'fall', 'crash', 'bearish', 'decline', 'down', 'negative', 'sell'
            ]
            
            keyword_count = sum(1 for word in strength_keywords if word in combined_text)
            
            # Adjust confidence based on keyword presence
            if keyword_count >= 3:
                confidence = min(0.85, confidence + 0.1)
            elif keyword_count >= 1:
                confidence = min(0.75, confidence + 0.05)
            
            narrative = (f"Market sentiment appears {sentiment_desc} based on {len(headlines)} "
                        f"recent headlines. Key sentiment indicators present: {keyword_count}")
            
            logger.info(f"📰 Local narrative: {sentiment_desc} (conf: {confidence:.2f})")
            
            return narrative, confidence
            
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return "Unable to analyze market sentiment", 0.0

# Global instance
local_sentiment = LocalSentimentAnalyzer()

def sentiment_via_local(headlines: List[str]) -> Tuple[float, Optional[str]]:
    """
    Drop-in replacement for sentiment_via_perplexity
    """
    return local_sentiment.analyze_headlines_sentiment(headlines)

def get_local_sentiment_analysis(headline: str) -> dict:
    """
    Drop-in replacement for get_sentiment_analysis from pplx_key_manager
    """
    try:
        sentiment, error = sentiment_via_local([headline])
        return {
            'success': error is None,
            'data': sentiment,
            'error': error
        }
    except Exception as e:
        return {
            'success': False, 
            'data': 0.5,
            'error': str(e)
        }

def get_local_narrative_summary(headlines: List[str]) -> dict:
    """
    Drop-in replacement for get_narrative_summary from pplx_key_manager
    """
    try:
        narrative, confidence = local_sentiment.get_narrative_summary(headlines)
        return {
            'success': True,
            'narrative': narrative,
            'confidence': confidence,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'narrative': "Analysis unavailable",
            'confidence': 0.0,
            'error': str(e)
        }

if __name__ == "__main__":
    # Test the local sentiment analyzer
    test_headlines = [
        "Bitcoin surges to new all-time high as institutional adoption continues",
        "Ethereum shows strong momentum with smart contract activity rising",
        "Crypto market experiences significant volatility amid regulatory concerns"
    ]
    
    print("🧪 Testing Local Sentiment Analyzer")
    print("=" * 50)
    
    sentiment, error = sentiment_via_local(test_headlines)
    print(f"Sentiment Score: {sentiment:.3f}")
    if error:
        print(f"Error: {error}")
    
    narrative_result = get_local_narrative_summary(test_headlines)
    print(f"Narrative: {narrative_result['narrative']}")
    print(f"Confidence: {narrative_result['confidence']:.3f}")
