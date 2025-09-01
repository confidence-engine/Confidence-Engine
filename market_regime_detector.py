import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class TrendRegime(Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways"
    DOWN = "down"
    STRONG_DOWN = "strong_down"

class LiquidityRegime(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class MomentumRegime(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class RegimeClassification:
    """Complete market regime classification"""
    volatility: VolatilityRegime
    trend: TrendRegime
    liquidity: LiquidityRegime
    momentum: MomentumRegime
    composite_score: float
    confidence: float

class MarketRegimeDetector:
    """
    Multi-dimensional market regime detection system
    """

    def __init__(self):
        self.volatility_thresholds = {
            'low': 0.02,      # 2% daily volatility
            'medium': 0.05,   # 5% daily volatility
            'high': 0.08,     # 8% daily volatility
        }

        self.trend_thresholds = {
            'strong': 0.03,   # 3% trend strength
            'weak': 0.01,     # 1% trend strength
        }

        self.momentum_thresholds = {
            'strong': 0.02,   # 2% momentum
            'weak': 0.005,    # 0.5% momentum
        }

    def classify_regime(self, data: Dict[str, pd.DataFrame]) -> Dict[str, RegimeClassification]:
        """
        Classify market regime for each asset
        """
        results = {}

        for asset, df in data.items():
            try:
                regime = self._classify_single_asset(df)
                results[asset] = regime
            except Exception as e:
                logger.warning(f"Regime classification failed for {asset}: {e}")
                # Return neutral regime on failure
                results[asset] = RegimeClassification(
                    volatility=VolatilityRegime.MEDIUM,
                    trend=TrendRegime.SIDEWAYS,
                    liquidity=LiquidityRegime.MEDIUM,
                    momentum=MomentumRegime.NEUTRAL,
                    composite_score=0.0,
                    confidence=0.5
                )

        return results

    def _classify_single_asset(self, df: pd.DataFrame) -> RegimeClassification:
        """Classify regime for a single asset"""

        # Calculate all regime indicators
        volatility_regime = self._classify_volatility(df)
        trend_regime = self._classify_trend(df)
        liquidity_regime = self._classify_liquidity(df)
        momentum_regime = self._classify_momentum(df)

        # Calculate composite score
        composite_score = self._calculate_composite_score(
            volatility_regime, trend_regime, liquidity_regime, momentum_regime
        )

        # Calculate confidence based on indicator agreement
        confidence = self._calculate_confidence(
            [volatility_regime, trend_regime, liquidity_regime, momentum_regime]
        )

        return RegimeClassification(
            volatility=volatility_regime,
            trend=trend_regime,
            liquidity=liquidity_regime,
            momentum=momentum_regime,
            composite_score=composite_score,
            confidence=confidence
        )

    def _classify_volatility(self, df: pd.DataFrame) -> VolatilityRegime:
        """Classify volatility regime"""
        try:
            # Calculate realized volatility (20-period rolling)
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()

            current_vol = volatility.iloc[-1]

            if current_vol <= self.volatility_thresholds['low']:
                return VolatilityRegime.LOW
            elif current_vol <= self.volatility_thresholds['medium']:
                return VolatilityRegime.MEDIUM
            elif current_vol <= self.volatility_thresholds['high']:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.EXTREME

        except Exception:
            return VolatilityRegime.MEDIUM

    def _classify_trend(self, df: pd.DataFrame) -> TrendRegime:
        """Classify trend regime"""
        try:
            # Use multiple timeframes for trend classification
            short_trend = self._calculate_trend_strength(df, 20)  # 20-period
            medium_trend = self._calculate_trend_strength(df, 50)  # 50-period
            long_trend = self._calculate_trend_strength(df, 100)  # 100-period

            # Weighted trend strength
            trend_strength = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)

            if trend_strength > self.trend_thresholds['strong']:
                return TrendRegime.STRONG_UP
            elif trend_strength > self.trend_thresholds['weak']:
                return TrendRegime.UP
            elif trend_strength < -self.trend_thresholds['strong']:
                return TrendRegime.STRONG_DOWN
            elif trend_strength < -self.trend_thresholds['weak']:
                return TrendRegime.DOWN
            else:
                return TrendRegime.SIDEWAYS

        except Exception:
            return TrendRegime.SIDEWAYS

    def _calculate_trend_strength(self, df: pd.DataFrame, period: int) -> float:
        """Calculate trend strength using linear regression slope"""
        try:
            prices = df['close'].tail(period)
            if len(prices) < period:
                return 0.0

            # Normalize prices to percentage change from start
            start_price = prices.iloc[0]
            normalized = (prices - start_price) / start_price

            # Calculate slope using linear regression
            x = np.arange(len(normalized))
            slope = np.polyfit(x, normalized.values, 1)[0]

            return slope

        except Exception:
            return 0.0

    def _classify_liquidity(self, df: pd.DataFrame) -> LiquidityRegime:
        """Classify liquidity regime based on volume patterns"""
        try:
            # Calculate volume relative to price (dollar volume)
            dollar_volume = df['close'] * df['volume']

            # Compare to historical average
            avg_volume = dollar_volume.rolling(20).mean()
            current_volume = dollar_volume.iloc[-1]
            avg_volume_val = avg_volume.iloc[-1]

            if avg_volume_val == 0:
                return LiquidityRegime.MEDIUM

            volume_ratio = current_volume / avg_volume_val

            if volume_ratio > 1.5:  # 50% above average
                return LiquidityRegime.HIGH
            elif volume_ratio > 0.7:  # 30% below average
                return LiquidityRegime.MEDIUM
            else:
                return LiquidityRegime.LOW

        except Exception:
            return LiquidityRegime.MEDIUM

    def _classify_momentum(self, df: pd.DataFrame) -> MomentumRegime:
        """Classify momentum regime"""
        try:
            # Calculate momentum using RSI and MACD
            rsi = self._calculate_rsi(df['close'])
            macd, signal = self._calculate_macd(df['close'])

            # Combine momentum indicators
            rsi_score = (rsi - 50) / 50  # Normalize RSI around 50
            macd_score = (macd - signal) / df['close'].rolling(20).std().iloc[-1]  # Normalize MACD

            momentum_score = (rsi_score * 0.6 + macd_score * 0.4)

            if momentum_score > self.momentum_thresholds['strong']:
                return MomentumRegime.STRONG_BULLISH
            elif momentum_score > self.momentum_thresholds['weak']:
                return MomentumRegime.BULLISH
            elif momentum_score < -self.momentum_thresholds['strong']:
                return MomentumRegime.STRONG_BEARISH
            elif momentum_score < -self.momentum_thresholds['weak']:
                return MomentumRegime.BEARISH
            else:
                return MomentumRegime.NEUTRAL

        except Exception:
            return MomentumRegime.NEUTRAL

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception:
            return 50.0

    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD"""
        try:
            ema12 = prices.ewm(span=12, adjust=False).mean()
            ema26 = prices.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            return macd.iloc[-1], signal.iloc[-1]
        except Exception:
            return 0.0, 0.0

    def _calculate_composite_score(self,
                                 vol_regime: VolatilityRegime,
                                 trend_regime: TrendRegime,
                                 liq_regime: LiquidityRegime,
                                 mom_regime: MomentumRegime) -> float:
        """Calculate composite regime score"""

        # Assign numerical scores to regimes
        vol_score = {
            VolatilityRegime.LOW: 0.2,
            VolatilityRegime.MEDIUM: 0.0,
            VolatilityRegime.HIGH: -0.2,
            VolatilityRegime.EXTREME: -0.4
        }.get(vol_regime, 0.0)

        trend_score = {
            TrendRegime.STRONG_UP: 0.4,
            TrendRegime.UP: 0.2,
            TrendRegime.SIDEWAYS: 0.0,
            TrendRegime.DOWN: -0.2,
            TrendRegime.STRONG_DOWN: -0.4
        }.get(trend_regime, 0.0)

        liq_score = {
            LiquidityRegime.HIGH: 0.1,
            LiquidityRegime.MEDIUM: 0.0,
            LiquidityRegime.LOW: -0.1
        }.get(liq_regime, 0.0)

        mom_score = {
            MomentumRegime.STRONG_BULLISH: 0.3,
            MomentumRegime.BULLISH: 0.15,
            MomentumRegime.NEUTRAL: 0.0,
            MomentumRegime.BEARISH: -0.15,
            MomentumRegime.STRONG_BEARISH: -0.3
        }.get(mom_regime, 0.0)

        # Weighted composite score
        composite = (
            vol_score * 0.2 +
            trend_score * 0.4 +
            liq_score * 0.1 +
            mom_score * 0.3
        )

        return np.clip(composite, -1.0, 1.0)

    def _calculate_confidence(self, regimes: List[Any]) -> float:
        """Calculate confidence based on regime agreement"""
        try:
            # Simple agreement-based confidence
            # In a more sophisticated implementation, this could use
            # statistical measures of regime stability
            return 0.8  # Placeholder - high confidence for now
        except Exception:
            return 0.5

    def get_regime_string(self, regime: RegimeClassification) -> str:
        """Convert regime classification to readable string"""
        return f"{regime.trend.value}_{regime.volatility.value}_{regime.momentum.value}"

    def is_bullish_regime(self, regime: RegimeClassification) -> bool:
        """Check if regime is generally bullish"""
        bullish_trends = [TrendRegime.UP, TrendRegime.STRONG_UP]
        bullish_momentum = [MomentumRegime.BULLISH, MomentumRegime.STRONG_BULLISH]

        return (
            regime.trend in bullish_trends and
            regime.momentum in bullish_momentum
        )

    def is_bearish_regime(self, regime: RegimeClassification) -> bool:
        """Check if regime is generally bearish"""
        bearish_trends = [TrendRegime.DOWN, TrendRegime.STRONG_DOWN]
        bearish_momentum = [MomentumRegime.BEARISH, MomentumRegime.STRONG_BEARISH]

        return (
            regime.trend in bearish_trends and
            regime.momentum in bearish_momentum
        )
