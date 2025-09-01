import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class EntrySignal:
    """Complete entry signal with all components"""
    direction: str  # 'buy' or 'sell'
    strength: SignalStrength
    confidence: float
    components: Dict[str, float]  # Individual signal components
    timestamp: pd.Timestamp
    asset: str

@dataclass
class ExitSignal:
    """Exit signal for position management"""
    reason: str  # 'take_profit', 'stop_loss', 'signal_reversal', 'timeout'
    confidence: float
    target_price: float
    timestamp: pd.Timestamp

class AdvancedEntryLogic:
    """
    Multi-timeframe, multi-signal entry logic with ML integration
    """

    def __init__(self, ensemble_model=None, risk_manager=None):
        self.ensemble_model = ensemble_model
        self.risk_manager = risk_manager

        # Signal weights
        self.signal_weights = {
            'technical': 0.25,
            'sentiment': 0.20,
            'momentum': 0.20,
            'volume': 0.15,
            'ml_score': 0.20
        }

        # Thresholds
        self.min_confidence_threshold = 0.65
        self.strong_signal_threshold = 0.80
        self.min_signals_required = 4  # Out of 5 signals

    def should_enter(self,
                    data: Dict[str, pd.DataFrame],
                    sentiment: float,
                    regime: str = 'trending') -> Optional[EntrySignal]:
        """
        Determine if we should enter a position based on all signals
        """

        signals = {}

        # Technical signals (multi-timeframe)
        signals['technical'] = self._technical_signal(data, regime)

        # Sentiment signal
        signals['sentiment'] = self._sentiment_signal(sentiment)

        # Momentum signal
        signals['momentum'] = self._momentum_signal(data)

        # Volume signal
        signals['volume'] = self._volume_signal(data)

        # ML signal
        signals['ml_score'] = self._ml_signal(data)

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(signals)

        # Determine direction and strength
        direction = self._determine_direction(signals)
        strength = self._determine_strength(signals, confidence)

        # Check entry conditions
        if self._meets_entry_criteria(signals, confidence):
            return EntrySignal(
                direction=direction,
                strength=strength,
                confidence=confidence,
                components=signals,
                timestamp=pd.Timestamp.now(),
                asset=list(data.keys())[0] if data else 'unknown'
            )

        return None

    def _technical_signal(self, data: Dict[str, pd.DataFrame], regime: str) -> float:
        """Calculate technical signal strength"""
        try:
            # Use primary asset for technical analysis
            primary_asset = list(data.keys())[0]
            df = data[primary_asset]

            # Multi-timeframe EMA signals
            ema_signals = self._calculate_ema_signals(df)

            # RSI signals
            rsi_signal = self._calculate_rsi_signal(df)

            # MACD signals
            macd_signal = self._calculate_macd_signal(df)

            # Bollinger Band signals
            bb_signal = self._calculate_bb_signal(df)

            # Combine technical signals
            technical_score = (
                ema_signals * 0.4 +
                rsi_signal * 0.25 +
                macd_signal * 0.20 +
                bb_signal * 0.15
            )

            return technical_score

        except Exception as e:
            logger.warning(f"Technical signal calculation failed: {e}")
            return 0.0

    def _calculate_ema_signals(self, df: pd.DataFrame) -> float:
        """Calculate EMA crossover signals"""
        try:
            # Multiple EMA combinations
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            ema50 = df['close'].ewm(span=50, adjust=False).mean()

            # EMA cross signals
            ema12_26_cross = 1 if ema12.iloc[-1] > ema26.iloc[-1] else -1
            ema12_50_cross = 1 if ema12.iloc[-1] > ema50.iloc[-1] else -1
            ema26_50_cross = 1 if ema26.iloc[-1] > ema50.iloc[-1] else -1

            # Trend strength
            trend_strength = (ema12.iloc[-1] - ema26.iloc[-1]) / df['close'].iloc[-1]

            # Combine signals
            ema_score = (
                ema12_26_cross * 0.4 +
                ema12_50_cross * 0.3 +
                ema26_50_cross * 0.3
            ) * (1 + trend_strength)

            return np.clip(ema_score, -1, 1)

        except Exception:
            return 0.0

    def _calculate_rsi_signal(self, df: pd.DataFrame) -> float:
        """Calculate RSI-based signal"""
        try:
            rsi = self._calculate_rsi(df['close'])

            # RSI signal: oversold (<30) = buy, overbought (>70) = sell
            if rsi < 30:
                return 0.8  # Strong buy signal
            elif rsi < 45:
                return 0.4  # Moderate buy signal
            elif rsi > 70:
                return -0.8  # Strong sell signal
            elif rsi > 55:
                return -0.4  # Moderate sell signal
            else:
                return 0.0  # Neutral

        except Exception:
            return 0.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _calculate_macd_signal(self, df: pd.DataFrame) -> float:
        """Calculate MACD signal"""
        try:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal

            # MACD signal based on histogram and crossovers
            hist_signal = np.sign(histogram.iloc[-1])
            macd_above_signal = 1 if macd.iloc[-1] > signal.iloc[-1] else -1

            macd_score = (hist_signal * 0.6 + macd_above_signal * 0.4)
            return macd_score

        except Exception:
            return 0.0

    def _calculate_bb_signal(self, df: pd.DataFrame) -> float:
        """Calculate Bollinger Band signal"""
        try:
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)

            current_price = df['close'].iloc[-1]

            # Bollinger Band position
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

            # Signal based on position and squeeze
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma20.iloc[-1]

            if bb_position < 0.2:  # Near lower band
                signal = 0.6
            elif bb_position > 0.8:  # Near upper band
                signal = -0.6
            else:
                signal = 0.0

            # Reduce signal strength in squeeze conditions
            if bb_width < 0.05:  # Squeeze condition
                signal *= 0.5

            return signal

        except Exception:
            return 0.0

    def _sentiment_signal(self, sentiment: float) -> float:
        """Convert sentiment score to signal"""
        # Sentiment is already in [-1, 1] range
        return sentiment

    def _momentum_signal(self, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate momentum-based signal"""
        try:
            primary_asset = list(data.keys())[0]
            df = data[primary_asset]

            # Multiple momentum indicators
            roc5 = df['close'].pct_change(5).iloc[-1]
            roc10 = df['close'].pct_change(10).iloc[-1]
            roc20 = df['close'].pct_change(20).iloc[-1]

            # Rate of change momentum
            momentum_score = (roc5 * 0.5 + roc10 * 0.3 + roc20 * 0.2)

            return np.clip(momentum_score * 2, -1, 1)  # Scale and clip

        except Exception:
            return 0.0

    def _volume_signal(self, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate volume-based signal"""
        try:
            primary_asset = list(data.keys())[0]
            df = data[primary_asset]

            # Volume relative to moving average
            vol_sma20 = df['volume'].rolling(20).mean()
            current_vol = df['volume'].iloc[-1]
            avg_vol = vol_sma20.iloc[-1]

            if avg_vol == 0:
                return 0.0

            vol_ratio = current_vol / avg_vol

            # Volume confirmation signal
            if vol_ratio > 1.5:
                return 0.7  # High volume confirmation
            elif vol_ratio > 1.2:
                return 0.4  # Moderate volume confirmation
            elif vol_ratio < 0.7:
                return -0.4  # Low volume caution
            else:
                return 0.0

        except Exception:
            return 0.0

    def _ml_signal(self, data: Dict[str, pd.DataFrame]) -> float:
        """Get ML model signal"""
        if self.ensemble_model is None:
            return 0.0

        try:
            # Extract features for ML model
            features = self._extract_ml_features(data)
            if features is not None:
                signal = self.ensemble_model.predict(features)
                return signal
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"ML signal failed: {e}")
            return 0.0

    def _extract_ml_features(self, data: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Extract features for ML model"""
        try:
            primary_asset = list(data.keys())[0]
            df = data[primary_asset]

            # This should match the feature engineering in your training pipeline
            features = []

            # Basic returns
            features.append(df['close'].pct_change().iloc[-1])

            # RSI
            rsi = self._calculate_rsi(df['close'])
            features.append(rsi / 100.0)  # Normalize to 0-1

            # MACD components
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            features.extend([macd.iloc[-1], signal.iloc[-1]])

            # Bollinger Bands
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            bb_position = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            features.append(bb_position)

            # Volume indicators
            vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            features.append(vol_ratio)

            return np.array(features)

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def _calculate_overall_confidence(self, signals: Dict[str, float]) -> float:
        """Calculate overall confidence from all signals"""
        try:
            weighted_sum = 0.0
            total_weight = 0.0

            for signal_name, signal_value in signals.items():
                weight = self.signal_weights.get(signal_name, 0.2)
                weighted_sum += abs(signal_value) * weight
                total_weight += weight

            if total_weight > 0:
                confidence = weighted_sum / total_weight
                return min(confidence, 1.0)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _determine_direction(self, signals: Dict[str, float]) -> str:
        """Determine overall signal direction"""
        try:
            # Use weighted average of signals
            weighted_sum = 0.0
            total_weight = 0.0

            for signal_name, signal_value in signals.items():
                weight = self.signal_weights.get(signal_name, 0.2)
                weighted_sum += signal_value * weight
                total_weight += weight

            return 'buy' if weighted_sum > 0 else 'sell'

        except Exception:
            return 'hold'

    def _determine_strength(self, signals: Dict[str, float], confidence: float) -> SignalStrength:
        """Determine signal strength"""
        try:
            # Count strong signals
            strong_signals = sum(1 for s in signals.values() if abs(s) > 0.7)

            if confidence > self.strong_signal_threshold and strong_signals >= 3:
                return SignalStrength.STRONG_BUY if self._determine_direction(signals) == 'buy' else SignalStrength.STRONG_SELL
            elif confidence > self.min_confidence_threshold:
                return SignalStrength.BUY if self._determine_direction(signals) == 'buy' else SignalStrength.SELL
            else:
                return SignalStrength.NEUTRAL

        except Exception:
            return SignalStrength.NEUTRAL

    def _meets_entry_criteria(self, signals: Dict[str, float], confidence: float) -> bool:
        """Check if signals meet entry criteria"""
        try:
            # Must have minimum confidence
            if confidence < self.min_confidence_threshold:
                return False

            # Must have minimum number of positive signals
            positive_signals = sum(1 for s in signals.values() if s > 0)
            if positive_signals < self.min_signals_required:
                return False

            # ML signal must be above threshold if available
            if 'ml_score' in signals and abs(signals['ml_score']) < 0.5:
                return False

            return True

        except Exception:
            return False

    def should_exit(self,
                   current_price: float,
                   entry_price: float,
                   stop_loss: float,
                   take_profit: float,
                   holding_time: int,
                   regime: str = 'trending') -> Optional[ExitSignal]:
        """
        Determine if we should exit a position
        """

        # Check stop loss
        if current_price <= stop_loss:
            return ExitSignal(
                reason='stop_loss',
                confidence=1.0,
                target_price=stop_loss,
                timestamp=pd.Timestamp.now()
            )

        # Check take profit
        if current_price >= take_profit:
            return ExitSignal(
                reason='take_profit',
                confidence=0.9,
                target_price=take_profit,
                timestamp=pd.Timestamp.now()
            )

        # Check for signal reversal (simplified)
        if self._detect_signal_reversal():
            return ExitSignal(
                reason='signal_reversal',
                confidence=0.7,
                target_price=current_price,
                timestamp=pd.Timestamp.now()
            )

        # Check timeout (24 hours = 86400 seconds)
        if holding_time > 86400:
            return ExitSignal(
                reason='timeout',
                confidence=0.5,
                target_price=current_price,
                timestamp=pd.Timestamp.now()
            )

        return None

    def _detect_signal_reversal(self) -> bool:
        """Detect if signals have reversed (simplified implementation)"""
        # This would typically analyze recent signal changes
        # For now, return False as a placeholder
        return False
