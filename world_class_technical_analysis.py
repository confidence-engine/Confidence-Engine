#!/usr/bin/env python3
"""
World-Class Technical Analysis Engine for Trading Agents
========================================================

This module provides professional-grade technical analysis for dynamic TP/SL calculation,
replacing all hardcoded percentage-based systems with market-adaptive risk management.

Features:
- ATR-based position sizing and targets
- RSI-based support/resistance detection  
- Bollinger Band dynamic targets
- Market regime detection (trending/ranging/volatile)
- Multi-timeframe confluence
- Confidence-based risk adjustment
- Robust error handling and fallbacks
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from enum import Enum
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down" 
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"

@dataclass
class TechnicalLevels:
    """Technical analysis levels and targets"""
    # ATR-based levels
    atr_value: float
    atr_sl_distance: float
    atr_tp_distance: float
    
    # RSI-based levels
    rsi_value: float
    rsi_support: Optional[float] = None
    rsi_resistance: Optional[float] = None
    
    # Bollinger Band levels
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    
    # Support/Resistance levels
    price_support: Optional[float] = None
    price_resistance: Optional[float] = None
    
    # Market regime
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = 0.0

@dataclass  
class RiskTargets:
    """Final risk management targets"""
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    position_size_multiplier: float = 1.0
    
    # Metadata
    sl_method: str = "unknown"
    tp_method: str = "unknown"
    confidence: float = 0.5
    risk_reward_ratio: float = 0.0
    
    def __post_init__(self):
        """Calculate risk/reward ratio"""
        if self.entry_price > 0 and self.stop_loss > 0 and self.take_profit_1 > 0:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit_1 - self.entry_price)
            self.risk_reward_ratio = reward / risk if risk > 0 else 0.0

class TechnicalAnalysisEngine:
    """
    Professional technical analysis engine for dynamic TP/SL calculation
    """
    
    def __init__(self, 
                 atr_period: int = 14,
                 rsi_period: int = 14, 
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 min_bars_required: int = 50):
        """
        Initialize the technical analysis engine
        
        Args:
            atr_period: Period for ATR calculation
            rsi_period: Period for RSI calculation  
            bb_period: Period for Bollinger Bands
            bb_std_dev: Standard deviation for Bollinger Bands
            min_bars_required: Minimum bars needed for analysis
        """
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.min_bars_required = min_bars_required
        
        # Regime detection parameters
        self.trend_lookback = 20
        self.volatility_lookback = 20
        
        # Risk management multipliers by regime (FIXED - was backwards)
        self.regime_multipliers = {
            MarketRegime.TRENDING_UP: {'sl': 1.5, 'tp': 3.5, 'size': 1.2},      # Strong trend = wider TP
            MarketRegime.TRENDING_DOWN: {'sl': 1.5, 'tp': 3.5, 'size': 1.2},    # Strong trend = wider TP  
            MarketRegime.RANGING: {'sl': 1.0, 'tp': 2.0, 'size': 1.0},          # Range = moderate targets
            MarketRegime.VOLATILE: {'sl': 1.2, 'tp': 3.0, 'size': 0.8},         # Volatile = wider TP, moderate SL
            MarketRegime.BREAKOUT: {'sl': 1.8, 'tp': 4.5, 'size': 1.3},         # Breakout = very wide TP
            MarketRegime.UNKNOWN: {'sl': 1.2, 'tp': 2.5, 'size': 0.8}           # Unknown = conservative
        }
    
    def calculate_atr(self, df: pd.DataFrame) -> float:
        """
        Calculate Average True Range with robust error handling
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            ATR value (normalized by current price)
        """
        try:
            if len(df) < self.atr_period:
                logger.warning(f"Insufficient data for ATR calculation: {len(df)} < {self.atr_period}")
                return 0.02  # 2% fallback
            
            # Calculate True Range components
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            # True Range is the maximum of the three
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # ATR is the moving average of True Range
            atr_raw = true_range.rolling(window=self.atr_period).mean().iloc[-1]
            
            # Normalize by current price
            current_price = df['close'].iloc[-1]
            atr_normalized = atr_raw / current_price if current_price > 0 else 0.02
            
            # Sanity checks
            if np.isnan(atr_normalized) or atr_normalized <= 0:
                logger.warning(f"Invalid ATR calculated: {atr_normalized}, using fallback")
                return 0.02
            
            # Cap extreme values
            atr_normalized = max(0.005, min(0.10, atr_normalized))  # 0.5% to 10%
            
            logger.debug(f"ATR calculated: {atr_normalized:.4f} ({atr_normalized*100:.2f}%)")
            return atr_normalized
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.02
    
    def calculate_rsi(self, df: pd.DataFrame) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Calculate RSI and identify support/resistance levels
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (current_rsi, support_level, resistance_level)
        """
        try:
            if len(df) < self.rsi_period + 10:
                logger.warning(f"Insufficient data for RSI calculation: {len(df)}")
                return 50.0, None, None
            
            # Calculate price changes
            delta = df['close'].diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=self.rsi_period).mean()
            avg_losses = losses.rolling(window=self.rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # Identify support and resistance levels from RSI
            support_level, resistance_level = self._find_rsi_support_resistance(df, rsi)
            
            logger.debug(f"RSI: {current_rsi:.2f}, Support: {support_level}, Resistance: {resistance_level}")
            return current_rsi, support_level, resistance_level
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0, None, None
    
    def _find_rsi_support_resistance(self, df: pd.DataFrame, rsi_series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """
        Find price levels corresponding to RSI support/resistance
        
        Args:
            df: Price DataFrame
            rsi_series: RSI values
            
        Returns:
            Tuple of (support_price, resistance_price)
        """
        try:
            if len(rsi_series) < 20:
                return None, None
            
            # Look for RSI levels where price bounced
            rsi_values = rsi_series.iloc[-50:].values  # Last 50 periods
            price_values = df['close'].iloc[-50:].values
            
            # Find RSI oversold bounces (potential support)
            oversold_mask = rsi_values < 35
            if np.any(oversold_mask):
                oversold_prices = price_values[oversold_mask]
                support_level = np.min(oversold_prices) if len(oversold_prices) > 0 else None
            else:
                support_level = None
            
            # Find RSI overbought rejections (potential resistance)  
            overbought_mask = rsi_values > 65
            if np.any(overbought_mask):
                overbought_prices = price_values[overbought_mask]
                resistance_level = np.max(overbought_prices) if len(overbought_prices) > 0 else None
            else:
                resistance_level = None
            
            return support_level, resistance_level
            
        except Exception as e:
            logger.error(f"Error finding RSI S/R levels: {e}")
            return None, None
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Calculate Bollinger Bands for dynamic targets
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band, band_width)
        """
        try:
            if len(df) < self.bb_period:
                logger.warning(f"Insufficient data for Bollinger Bands: {len(df)}")
                return None, None, None, None
            
            # Calculate moving average (middle band)
            sma = df['close'].rolling(window=self.bb_period).mean()
            
            # Calculate standard deviation
            std = df['close'].rolling(window=self.bb_period).std()
            
            # Calculate bands
            upper_band = sma + (std * self.bb_std_dev)
            lower_band = sma - (std * self.bb_std_dev)
            
            # Get current values
            current_upper = upper_band.iloc[-1]
            current_middle = sma.iloc[-1] 
            current_lower = lower_band.iloc[-1]
            
            # Calculate band width (normalized)
            current_price = df['close'].iloc[-1]
            band_width = (current_upper - current_lower) / current_price if current_price > 0 else 0
            
            logger.debug(f"BB: Upper={current_upper:.2f}, Middle={current_middle:.2f}, Lower={current_lower:.2f}, Width={band_width:.4f}")
            return current_upper, current_middle, current_lower, band_width
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return None, None, None, None
    
    def detect_market_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime using multiple indicators
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (regime, confidence)
        """
        try:
            if len(df) < self.trend_lookback:
                return MarketRegime.UNKNOWN, 0.0
            
            # Calculate trend indicators
            trend_score = self._calculate_trend_score(df)
            volatility_score = self._calculate_volatility_score(df) 
            volume_score = self._calculate_volume_score(df)
            
            # Combine scores to determine regime
            if abs(trend_score) > 0.6:
                if trend_score > 0:
                    regime = MarketRegime.TRENDING_UP
                else:
                    regime = MarketRegime.TRENDING_DOWN
                confidence = min(abs(trend_score), 0.9)
            elif volatility_score > 0.7:
                regime = MarketRegime.VOLATILE
                confidence = min(volatility_score, 0.9)
            elif volatility_score < 0.3 and abs(trend_score) < 0.3:
                regime = MarketRegime.RANGING
                confidence = 1.0 - max(volatility_score, abs(trend_score))
            else:
                regime = MarketRegime.UNKNOWN
                confidence = 0.5
            
            logger.debug(f"Market regime: {regime.value}, confidence: {confidence:.2f}")
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNKNOWN, 0.0
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend strength score (-1 to +1)"""
        try:
            # Use multiple EMAs for trend detection
            ema_short = df['close'].ewm(span=12).mean()
            ema_medium = df['close'].ewm(span=26).mean()
            ema_long = df['close'].ewm(span=50).mean()
            
            # Calculate trend alignment score
            current_price = df['close'].iloc[-1]
            short_val = ema_short.iloc[-1]
            medium_val = ema_medium.iloc[-1]
            long_val = ema_long.iloc[-1]
            
            # Score based on EMA alignment
            score = 0.0
            if current_price > short_val > medium_val > long_val:
                score = 0.8  # Strong uptrend
            elif current_price > short_val > medium_val:
                score = 0.5  # Medium uptrend
            elif current_price < short_val < medium_val < long_val:
                score = -0.8  # Strong downtrend
            elif current_price < short_val < medium_val:
                score = -0.5  # Medium downtrend
            
            # Adjust for recent momentum
            momentum = (current_price - df['close'].iloc[-self.trend_lookback]) / df['close'].iloc[-self.trend_lookback]
            score += momentum * 0.3  # Add momentum component
            
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate volatility score (0 to 1)"""
        try:
            # Calculate recent volatility
            returns = df['close'].pct_change().dropna()
            recent_vol = returns.iloc[-self.volatility_lookback:].std()
            
            # Calculate historical volatility for comparison
            historical_vol = returns.std()
            
            # Normalize volatility score
            vol_score = recent_vol / historical_vol if historical_vol > 0 else 1.0
            
            return max(0.0, min(1.0, vol_score))
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {e}")
            return 0.5
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume score (0 to 1)"""
        try:
            if 'volume' not in df.columns:
                return 0.5  # Neutral if no volume data
            
            # Calculate recent vs average volume
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].iloc[-50:].mean()
            
            vol_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            return max(0.0, min(1.0, vol_ratio))
            
        except Exception as e:
            logger.error(f"Error calculating volume score: {e}")
            return 0.5

    def analyze_technical_levels(self, df: pd.DataFrame) -> TechnicalLevels:
        """
        Perform comprehensive technical analysis
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            TechnicalLevels object with all analysis results
        """
        try:
            # Calculate ATR
            atr_value = self.calculate_atr(df)
            current_price = df['close'].iloc[-1]
            
            # Calculate RSI and S/R levels
            rsi_value, rsi_support, rsi_resistance = self.calculate_rsi(df)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bollinger_bands(df)
            
            # Detect market regime
            regime, regime_confidence = self.detect_market_regime(df)
            
            # Calculate price-based support/resistance
            price_support, price_resistance = self._find_price_support_resistance(df)
            
            # Get regime multipliers
            regime_mult = self.regime_multipliers.get(regime, self.regime_multipliers[MarketRegime.UNKNOWN])
            
            # Calculate ATR-based distances
            atr_sl_distance = atr_value * regime_mult['sl']
            atr_tp_distance = atr_value * regime_mult['tp']
            
            return TechnicalLevels(
                atr_value=atr_value,
                atr_sl_distance=atr_sl_distance,
                atr_tp_distance=atr_tp_distance,
                rsi_value=rsi_value,
                rsi_support=rsi_support,
                rsi_resistance=rsi_resistance,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_width=bb_width,
                price_support=price_support,
                price_resistance=price_resistance,
                regime=regime,
                regime_confidence=regime_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            # Return minimal fallback analysis
            return TechnicalLevels(
                atr_value=0.02,
                atr_sl_distance=0.025,
                atr_tp_distance=0.05,
                rsi_value=50.0
            )
    
    def _find_price_support_resistance(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Find price-based support and resistance levels
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        try:
            if len(df) < 20:
                return None, None
            
            # Look at recent swing highs and lows
            highs = df['high'].iloc[-50:]
            lows = df['low'].iloc[-50:]
            
            # Find recent swing points
            from scipy.signal import argrelextrema
            
            # Find local maxima and minima
            high_indices = argrelextrema(highs.values, np.greater, order=3)[0]
            low_indices = argrelextrema(lows.values, np.less, order=3)[0]
            
            # Get resistance from recent highs
            if len(high_indices) > 0:
                recent_highs = highs.iloc[high_indices[-3:]].values  # Last 3 swing highs
                resistance_level = np.max(recent_highs)
            else:
                resistance_level = None
            
            # Get support from recent lows  
            if len(low_indices) > 0:
                recent_lows = lows.iloc[low_indices[-3:]].values  # Last 3 swing lows
                support_level = np.min(recent_lows)
            else:
                support_level = None
            
            return support_level, resistance_level
            
        except Exception as e:
            logger.error(f"Error finding price S/R levels: {e}")
            return None, None
    
    def calculate_world_class_targets(self, 
                                    df: pd.DataFrame,
                                    side: str,
                                    confidence: float = 0.7,
                                    symbol: str = "Unknown") -> RiskTargets:
        """
        Calculate world-class TP/SL targets using comprehensive technical analysis
        
        Args:
            df: DataFrame with OHLC data
            side: 'buy' or 'sell'
            confidence: Trade confidence (0.0 to 1.0)
            symbol: Trading symbol for logging
            
        Returns:
            RiskTargets object with calculated targets
        """
        try:
            logger.info(f"Calculating world-class targets for {symbol}, side: {side}, confidence: {confidence:.2f}")
            
            # Perform comprehensive technical analysis
            levels = self.analyze_technical_levels(df)
            
            # Get current price
            entry_price = df['close'].iloc[-1]
            
            # Calculate targets based on side
            if side.lower() in ['buy', 'long']:
                stop_loss, take_profit_1, take_profit_2, sl_method, tp_method, size_mult = self._calculate_long_targets(
                    entry_price, levels, confidence
                )
            else:
                stop_loss, take_profit_1, take_profit_2, sl_method, tp_method, size_mult = self._calculate_short_targets(
                    entry_price, levels, confidence  
                )
            
            # Create and return risk targets
            targets = RiskTargets(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                position_size_multiplier=size_mult,
                sl_method=sl_method,
                tp_method=tp_method,
                confidence=confidence
            )
            
            logger.info(f"Targets calculated - SL: {stop_loss:.4f} ({sl_method}), "
                       f"TP1: {take_profit_1:.4f} ({tp_method}), "
                       f"R/R: {targets.risk_reward_ratio:.2f}")
            
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating world-class targets: {e}")
            return self._get_fallback_targets(df, side, confidence)
    
    def _calculate_long_targets(self, entry_price: float, levels: TechnicalLevels, confidence: float) -> Tuple[float, float, Optional[float], str, str, float]:
        """Calculate targets for long positions"""
        
        # Calculate stop loss using multiple methods
        sl_candidates = []
        sl_methods = []
        
        # ATR-based stop loss
        atr_sl = entry_price - (entry_price * levels.atr_sl_distance)
        sl_candidates.append(atr_sl)
        sl_methods.append("ATR")
        
        # Support-based stop loss
        if levels.price_support and levels.price_support < entry_price:
            support_sl = levels.price_support * 0.99  # Slightly below support
            sl_candidates.append(support_sl)
            sl_methods.append("Support")
        
        if levels.rsi_support and levels.rsi_support < entry_price:
            rsi_sl = levels.rsi_support * 0.995  # Slightly below RSI support
            sl_candidates.append(rsi_sl)
            sl_methods.append("RSI_Support")
        
        # Bollinger Band lower band
        if levels.bb_lower and levels.bb_lower < entry_price:
            bb_sl = levels.bb_lower * 0.99
            sl_candidates.append(bb_sl)
            sl_methods.append("BB_Lower")
        
        # Choose stop loss (use the closest to entry that's not too tight)
        min_sl_distance = entry_price * 0.005  # Minimum 0.5% stop
        valid_sls = [(sl, method) for sl, method in zip(sl_candidates, sl_methods) 
                     if entry_price - sl >= min_sl_distance]
        
        if valid_sls:
            # Choose the highest valid stop loss (least risk)
            stop_loss, sl_method = max(valid_sls, key=lambda x: x[0])
        else:
            # Fallback to ATR
            stop_loss, sl_method = atr_sl, "ATR_Fallback"
        
        # Calculate take profit using multiple methods
        tp_candidates = []
        tp_methods = []
        
        # ATR-based take profit
        atr_tp = entry_price + (entry_price * levels.atr_tp_distance)
        tp_candidates.append(atr_tp)
        tp_methods.append("ATR")
        
        # Resistance-based take profit
        if levels.price_resistance and levels.price_resistance > entry_price:
            resistance_tp = levels.price_resistance * 0.99  # Slightly below resistance
            tp_candidates.append(resistance_tp)
            tp_methods.append("Resistance")
        
        if levels.rsi_resistance and levels.rsi_resistance > entry_price:
            rsi_tp = levels.rsi_resistance * 0.995
            tp_candidates.append(rsi_tp)
            tp_methods.append("RSI_Resistance")
        
        # Bollinger Band upper band
        if levels.bb_upper and levels.bb_upper > entry_price:
            bb_tp = levels.bb_upper * 0.99
            tp_candidates.append(bb_tp)
            tp_methods.append("BB_Upper")
        
        # Choose take profit (prefer closest reasonable target)
        min_tp_distance = entry_price * 0.01  # Minimum 1% profit
        valid_tps = [(tp, method) for tp, method in zip(tp_candidates, tp_methods)
                     if tp - entry_price >= min_tp_distance]
        
        if valid_tps:
            # For high confidence trades, use more aggressive targets
            if confidence > 0.8:
                take_profit_1, tp_method = max(valid_tps, key=lambda x: x[0])
            else:
                # For lower confidence, use more conservative targets
                take_profit_1, tp_method = min(valid_tps, key=lambda x: x[0])
        else:
            take_profit_1, tp_method = atr_tp, "ATR_Fallback"
        
        # CRITICAL: Ensure minimum Risk/Reward ratio of 1.5:1
        risk_amount = entry_price - stop_loss
        reward_amount = take_profit_1 - entry_price
        current_rr = reward_amount / risk_amount if risk_amount > 0 else 0
        
        if current_rr < 1.5:
            # Adjust TP to meet minimum R/R requirement
            min_reward_needed = risk_amount * 1.5
            take_profit_1 = entry_price + min_reward_needed
            tp_method = f"{tp_method}_RR_Adjusted"
            logger.info(f"🔧 Adjusted TP for minimum 1.5:1 R/R: ${take_profit_1:.2f}")
        
        # Optional second take profit (extend target for high confidence)
        take_profit_2 = None
        if confidence > 0.75 and levels.regime in [MarketRegime.TRENDING_UP, MarketRegime.BREAKOUT]:
            extension_mult = 1.8 if confidence > 0.9 else 1.5
            take_profit_2 = entry_price + ((take_profit_1 - entry_price) * extension_mult)
        
        # Position size multiplier based on regime and confidence
        regime_mult = self.regime_multipliers.get(levels.regime, self.regime_multipliers[MarketRegime.UNKNOWN])
        confidence_mult = 0.7 + (confidence * 0.6)  # 0.7 to 1.3 based on confidence
        size_mult = regime_mult['size'] * confidence_mult
        
        return stop_loss, take_profit_1, take_profit_2, sl_method, tp_method, size_mult
    
    def _calculate_short_targets(self, entry_price: float, levels: TechnicalLevels, confidence: float) -> Tuple[float, float, Optional[float], str, str, float]:
        """Calculate targets for short positions"""
        
        # Calculate stop loss using multiple methods
        sl_candidates = []
        sl_methods = []
        
        # ATR-based stop loss  
        atr_sl = entry_price + (entry_price * levels.atr_sl_distance)
        sl_candidates.append(atr_sl)
        sl_methods.append("ATR")
        
        # Resistance-based stop loss
        if levels.price_resistance and levels.price_resistance > entry_price:
            resistance_sl = levels.price_resistance * 1.01  # Slightly above resistance
            sl_candidates.append(resistance_sl)
            sl_methods.append("Resistance")
        
        if levels.rsi_resistance and levels.rsi_resistance > entry_price:
            rsi_sl = levels.rsi_resistance * 1.005
            sl_candidates.append(rsi_sl)
            sl_methods.append("RSI_Resistance")
        
        # Bollinger Band upper band
        if levels.bb_upper and levels.bb_upper > entry_price:
            bb_sl = levels.bb_upper * 1.01
            sl_candidates.append(bb_sl)
            sl_methods.append("BB_Upper")
        
        # Choose stop loss (use the closest to entry that's not too tight)
        min_sl_distance = entry_price * 0.005  # Minimum 0.5% stop
        valid_sls = [(sl, method) for sl, method in zip(sl_candidates, sl_methods)
                     if sl - entry_price >= min_sl_distance]
        
        if valid_sls:
            # Choose the lowest valid stop loss (least risk)
            stop_loss, sl_method = min(valid_sls, key=lambda x: x[0])
        else:
            stop_loss, sl_method = atr_sl, "ATR_Fallback"
        
        # Calculate take profit using multiple methods
        tp_candidates = []
        tp_methods = []
        
        # ATR-based take profit
        atr_tp = entry_price - (entry_price * levels.atr_tp_distance)
        tp_candidates.append(atr_tp)
        tp_methods.append("ATR")
        
        # Support-based take profit
        if levels.price_support and levels.price_support < entry_price:
            support_tp = levels.price_support * 1.01  # Slightly above support
            tp_candidates.append(support_tp)
            tp_methods.append("Support")
        
        if levels.rsi_support and levels.rsi_support < entry_price:
            rsi_tp = levels.rsi_support * 1.005
            tp_candidates.append(rsi_tp)
            tp_methods.append("RSI_Support")
        
        # Bollinger Band lower band
        if levels.bb_lower and levels.bb_lower < entry_price:
            bb_tp = levels.bb_lower * 1.01
            tp_candidates.append(bb_tp)
            tp_methods.append("BB_Lower")
        
        # Choose take profit
        min_tp_distance = entry_price * 0.01  # Minimum 1% profit
        valid_tps = [(tp, method) for tp, method in zip(tp_candidates, tp_methods)
                     if entry_price - tp >= min_tp_distance]
        
        if valid_tps:
            if confidence > 0.8:
                take_profit_1, tp_method = min(valid_tps, key=lambda x: x[0])  # More aggressive
            else:
                take_profit_1, tp_method = max(valid_tps, key=lambda x: x[0])  # More conservative
        else:
            take_profit_1, tp_method = atr_tp, "ATR_Fallback"
        
        # CRITICAL: Ensure minimum Risk/Reward ratio of 1.5:1 for shorts
        risk_amount = stop_loss - entry_price
        reward_amount = entry_price - take_profit_1
        current_rr = reward_amount / risk_amount if risk_amount > 0 else 0
        
        if current_rr < 1.5:
            # Adjust TP to meet minimum R/R requirement
            min_reward_needed = risk_amount * 1.5
            take_profit_1 = entry_price - min_reward_needed
            tp_method = f"{tp_method}_RR_Adjusted"
            logger.info(f"🔧 Adjusted short TP for minimum 1.5:1 R/R: ${take_profit_1:.2f}")
        
        # Optional second take profit
        take_profit_2 = None
        if confidence > 0.75 and levels.regime in [MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT]:
            extension_mult = 1.8 if confidence > 0.9 else 1.5
            take_profit_2 = entry_price - ((entry_price - take_profit_1) * extension_mult)
        
        # Position size multiplier
        regime_mult = self.regime_multipliers.get(levels.regime, self.regime_multipliers[MarketRegime.UNKNOWN])
        confidence_mult = 0.7 + (confidence * 0.6)
        size_mult = regime_mult['size'] * confidence_mult
        
        return stop_loss, take_profit_1, take_profit_2, sl_method, tp_method, size_mult
    
    def _get_fallback_targets(self, df: pd.DataFrame, side: str, confidence: float) -> RiskTargets:
        """Fallback targets when full analysis fails"""
        try:
            entry_price = df['close'].iloc[-1]
            
            # Simple ATR-based fallback
            atr = self.calculate_atr(df)
            
            if side.lower() in ['buy', 'long']:
                stop_loss = entry_price * (1 - atr * 1.5)
                take_profit_1 = entry_price * (1 + atr * 2.5)
            else:
                stop_loss = entry_price * (1 + atr * 1.5)
                take_profit_1 = entry_price * (1 - atr * 2.5)
            
            return RiskTargets(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                position_size_multiplier=0.8,  # Conservative sizing for fallback
                sl_method="ATR_Fallback",
                tp_method="ATR_Fallback",
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in fallback targets: {e}")
            # Last resort hardcoded fallback
            entry_price = df['close'].iloc[-1]
            if side.lower() in ['buy', 'long']:
                return RiskTargets(
                    entry_price=entry_price,
                    stop_loss=entry_price * 0.97,  # 3% SL
                    take_profit_1=entry_price * 1.06,  # 6% TP
                    position_size_multiplier=0.5,
                    sl_method="Emergency_Fallback",
                    tp_method="Emergency_Fallback",
                    confidence=confidence
                )
            else:
                return RiskTargets(
                    entry_price=entry_price,
                    stop_loss=entry_price * 1.03,  # 3% SL
                    take_profit_1=entry_price * 0.94,  # 6% TP
                    position_size_multiplier=0.5,
                    sl_method="Emergency_Fallback", 
                    tp_method="Emergency_Fallback",
                    confidence=confidence
                )


# Utility functions for backward compatibility and ease of use
def calculate_world_class_crypto_targets(df: pd.DataFrame, 
                                       side: str, 
                                       confidence: float = 0.7,
                                       symbol: str = "Unknown") -> Dict[str, float]:
    """
    Convenience function for calculating world-class targets
    Returns dictionary for backward compatibility
    """
    engine = TechnicalAnalysisEngine()
    targets = engine.calculate_world_class_targets(df, side, confidence, symbol)
    
    return {
        'tp_price': targets.take_profit_1,
        'sl_price': targets.stop_loss,
        'tp_pct': abs(targets.take_profit_1 - targets.entry_price) / targets.entry_price,
        'sl_pct': abs(targets.stop_loss - targets.entry_price) / targets.entry_price,
        'position_size_multiplier': targets.position_size_multiplier,
        'risk_reward_ratio': targets.risk_reward_ratio,
        'confidence': targets.confidence,
        'method': f"{targets.sl_method}|{targets.tp_method}",
        'tp_price_2': targets.take_profit_2
    }


if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    # Generate realistic OHLC data
    returns = np.random.normal(0, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.002, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 100)
    })
    
    # Test the engine
    engine = TechnicalAnalysisEngine()
    
    print("Testing Technical Analysis Engine")
    print("=" * 50)
    
    # Test technical analysis
    levels = engine.analyze_technical_levels(sample_data)
    print(f"ATR: {levels.atr_value:.4f}")
    print(f"RSI: {levels.rsi_value:.2f}")
    print(f"Market Regime: {levels.regime.value}")
    print(f"BB Width: {levels.bb_width:.4f}" if levels.bb_width else "BB Width: None")
    
    # Test target calculation
    targets = engine.calculate_world_class_targets(sample_data, 'buy', confidence=0.8, symbol='TEST')
    print(f"\nTargets for BUY:")
    print(f"Entry: ${targets.entry_price:.2f}")
    print(f"Stop Loss: ${targets.stop_loss:.2f} ({targets.sl_method})")
    print(f"Take Profit: ${targets.take_profit_1:.2f} ({targets.tp_method})")
    print(f"Risk/Reward: {targets.risk_reward_ratio:.2f}")
    print(f"Position Size Mult: {targets.position_size_multiplier:.2f}")
    
    print("\nEngine testing completed successfully!")
