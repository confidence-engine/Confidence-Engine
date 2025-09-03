def compute(narrative_score: float, price_score: float) -> float:
    return narrative_score - price_score

def reason(divergence_value: float, conf: float, vol_z: float, thresh: float, conf_cut: float) -> str:
    if conf < conf_cut:
        return "LOW_CONFIDENCE"
    if vol_z < -0.5:
        return "WEAK_VOLUME"
    if abs(divergence_value) < thresh:
        return "SMALL_DIVERGENCE"
    return "NARR_LEADS_PRICE" if divergence_value > 0 else "PRICE_LEADS_NARR"

def calculate_signal_quality(sentiment_score: float, price_momentum: float, volume_z_score: float, 
                           news_volume: int = 5, rsi: float = 50.0) -> float:
    """
    Calculate signal quality score from 0-10
    Higher score = higher conviction trade
    
    Args:
        sentiment_score: Normalized sentiment (-1 to 1)
        price_momentum: Price momentum score (-1 to 1) 
        volume_z_score: Volume Z-score
        news_volume: Number of news items contributing to sentiment
        rsi: RSI value (0-100)
    
    Returns:
        Quality score from 0-10
    """
    quality = 0.0
    
    # Sentiment strength (0-4 points)
    sentiment_abs = abs(sentiment_score)
    if sentiment_abs > 0.8: 
        quality += 4.0
    elif sentiment_abs > 0.6: 
        quality += 3.0
    elif sentiment_abs > 0.4: 
        quality += 2.0
    elif sentiment_abs > 0.2: 
        quality += 1.0
    
    # Price momentum clarity (0-3 points)
    price_abs = abs(price_momentum)
    if price_abs > 0.8: 
        quality += 3.0
    elif price_abs > 0.6: 
        quality += 2.0
    elif price_abs > 0.4: 
        quality += 1.0
    
    # Volume confirmation (0-2 points)
    if volume_z_score > 1.5: 
        quality += 2.0
    elif volume_z_score > 0.5: 
        quality += 1.0
    
    # RSI extremes bonus (0-1 points)
    # Strong signals when RSI is oversold/overbought
    if rsi > 70 or rsi < 30:
        quality += 1.0
    elif rsi > 60 or rsi < 40:
        quality += 0.5
    
    # Divergence type bonus - contrarian trades in extreme conditions get boost
    is_contrarian = (sentiment_score > 0 and price_momentum < 0) or (sentiment_score < 0 and price_momentum > 0)
    is_momentum = (sentiment_score > 0 and price_momentum > 0) or (sentiment_score < 0 and price_momentum < 0)
    
    if is_contrarian:
        # Contrarian trades stronger with fewer headlines (less noise)
        if news_volume <= 3:
            quality += 0.5
        # Extra boost for contrarian in extreme RSI conditions
        if (rsi > 75 and sentiment_score < 0) or (rsi < 25 and sentiment_score > 0):
            quality += 0.5
    elif is_momentum:
        # Momentum trades stronger with more headlines (confirmation)
        if news_volume >= 8:
            quality += 0.5
    
    return min(quality, 10.0)

def calculate_conviction_score(signal_quality: float, regime_alignment: float, 
                             volatility_score: float, confirmation_score: float) -> float:
    """
    Calculate overall conviction score (1-10) combining all factors
    
    Args:
        signal_quality: Signal quality score (0-10)
        regime_alignment: Market regime alignment (0-1)
        volatility_score: Volatility favorability (0-1) 
        confirmation_score: Price confirmation (0-1)
    
    Returns:
        Conviction score from 1-10
    """
    # Weighted combination
    quality_points = (signal_quality / 10.0) * 4.0    # 40% weight (0-4 points)
    regime_points = regime_alignment * 3.0              # 30% weight (0-3 points)
    vol_points = volatility_score * 2.0                 # 20% weight (0-2 points)
    conf_points = confirmation_score * 1.0              # 10% weight (0-1 points)
    
    total_conviction = quality_points + regime_points + vol_points + conf_points
    
    return max(1.0, min(total_conviction, 10.0))
