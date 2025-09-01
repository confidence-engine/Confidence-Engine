import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Dynamic risk limits based on market conditions"""
    portfolio_var_limit: float = 0.02  # 2% portfolio VaR
    max_drawdown_limit: float = 0.05   # 5% max drawdown
    max_single_position: float = 0.10  # 10% max single position
    max_correlation: float = 0.7       # Maximum allowed correlation
    min_diversification: int = 3       # Minimum assets in portfolio

@dataclass
class RegimeAdjustment:
    """Risk adjustments based on market regime"""
    volatility_multiplier: float
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float

class KellyPositionSizer:
    """
    Kelly Criterion position sizing with safeguards
    """
    
    def __init__(self):
        self.max_kelly_fraction = 0.25  # Cap at 25% of portfolio
        self.min_kelly_fraction = 0.01  # Minimum 1% of portfolio
        
    def calculate_kelly_size(self, win_probability: float, win_loss_ratio: float, 
                           portfolio_value: float, regime: str = 'normal') -> float:
        """
        Calculate position size using Kelly Criterion
        
        Kelly Formula: f = (bp - q) / b
        where:
        - f = fraction of capital to wager
        - b = odds received on the wager (win_loss_ratio)
        - p = probability of winning (win_probability)
        - q = probability of losing (1 - win_probability)
        """
        try:
            # Kelly calculation
            p = max(0.01, min(0.99, win_probability))  # Clamp probability
            q = 1 - p
            b = max(0.1, win_loss_ratio)  # Minimum ratio
            
            kelly_fraction = (b * p - q) / b
            
            # Apply regime adjustments
            if regime == 'high_volatility':
                kelly_fraction *= 0.5  # More conservative in volatile markets
            elif regime == 'low_volatility':
                kelly_fraction *= 1.2  # Slightly more aggressive in stable markets
            
            # Apply safeguards
            kelly_fraction = max(self.min_kelly_fraction, 
                               min(self.max_kelly_fraction, kelly_fraction))
            
            position_size = portfolio_value * kelly_fraction
            
            logger.debug(f"Kelly sizing: p={p:.3f}, b={b:.2f}, kelly_frac={kelly_fraction:.3f}, size=${position_size:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.warning(f"Kelly calculation failed: {e}, using default 1%")
            return portfolio_value * 0.01

class AdvancedRiskManager:
    """
    Advanced risk management with dynamic sizing, regime adjustments,
    and portfolio-level controls
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/risk_config.json"
        self.risk_limits = RiskLimits()
        self.regime_adjustments = self._load_regime_adjustments()
        self.portfolio_state = {}
        self.correlation_matrix = {}
        self.var_history = []
        
        # Initialize Kelly position sizer
        self.kelly_sizer = KellyPositionSizer()

    def _load_regime_adjustments(self) -> Dict[str, RegimeAdjustment]:
        """Load regime-based risk adjustments"""
        return {
            'high_volatility': RegimeAdjustment(
                volatility_multiplier=1.5,
                position_size_multiplier=0.3,
                stop_loss_multiplier=1.5,
                take_profit_multiplier=0.7
            ),
            'low_volatility': RegimeAdjustment(
                volatility_multiplier=0.7,
                position_size_multiplier=1.2,
                stop_loss_multiplier=0.8,
                take_profit_multiplier=1.3
            ),
            'trending': RegimeAdjustment(
                volatility_multiplier=1.0,
                position_size_multiplier=1.0,
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.0
            ),
            'sideways': RegimeAdjustment(
                volatility_multiplier=1.2,
                position_size_multiplier=0.7,
                stop_loss_multiplier=1.2,
                take_profit_multiplier=0.8
            ),
            'bull_market': RegimeAdjustment(
                volatility_multiplier=0.8,
                position_size_multiplier=1.1,
                stop_loss_multiplier=0.9,
                take_profit_multiplier=1.2
            ),
            'bear_market': RegimeAdjustment(
                volatility_multiplier=1.3,
                position_size_multiplier=0.5,
                stop_loss_multiplier=1.3,
                take_profit_multiplier=0.6
            )
        }

    def calculate_portfolio_var(self, positions: Dict[str, float], prices: Dict[str, float]) -> float:
        """Calculate portfolio Value at Risk using historical simulation"""
        try:
            # Simplified VaR calculation - in production use full historical simulation
            portfolio_value = sum(positions.get(asset, 0) * prices.get(asset, 0) for asset in positions)

            if portfolio_value == 0:
                return 0.0

            # For small portfolios or initial positions, use conservative estimate
            if len([asset for asset, size in positions.items() if size > 0]) <= 2:
                return 0.01  # 1% VaR for small portfolios

            # Use recent volatility as proxy for VaR
            returns = []
            for asset in positions:
                if asset in self.portfolio_state:
                    asset_returns = self.portfolio_state[asset].get('returns', [])
                    if asset_returns:
                        returns.extend(asset_returns[-30:])  # Last 30 periods

            if not returns:
                return 0.01  # Conservative 1% VaR for new positions

            var_95 = np.percentile(returns, 5)  # 95% confidence VaR
            return min(abs(var_95), 0.015)  # Cap at 1.5% to allow initial trading

        except Exception as e:
            logger.warning(f"VaR calculation failed: {e}")
            return 0.01  # Conservative default

    def get_regime_adjustment(self, regime: str) -> RegimeAdjustment:
        """Get risk adjustment for current market regime"""
        return self.regime_adjustments.get(regime, self.regime_adjustments['trending'])

    def calculate_kelly_size(self,
                           win_probability: float,
                           win_loss_ratio: float,
                           current_portfolio: float,
                           regime: str = 'trending') -> float:
        """Calculate Kelly-optimized position size with regime adjustments"""

        # Kelly formula: f = (bp - q) / b
        # where: b = odds (win_loss_ratio), p = win prob, q = loss prob
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio

        # Apply half-Kelly for conservatism
        optimal_size = kelly_fraction * 0.5

        # Get regime adjustment
        regime_adj = self.get_regime_adjustment(regime)
        optimal_size *= regime_adj.position_size_multiplier

        # Apply constraints
        max_size = min(
            current_portfolio * self.risk_limits.max_single_position,
            optimal_size
        )

        return max(0.0, min(max_size, current_portfolio * 0.05))  # Cap at 5%

    def check_portfolio_limits(self, positions: Dict[str, float], prices: Dict[str, float], 
                             new_position: Optional[Tuple[str, float]] = None) -> Dict[str, bool]:
        """Check if adding a new position would violate portfolio limits"""

        # Calculate current portfolio value
        current_value = sum(positions.get(asset, 0) * prices.get(asset, 0) for asset in positions)

        # Add new position if provided
        test_positions = positions.copy()
        if new_position:
            asset, size = new_position
            test_positions[asset] = test_positions.get(asset, 0) + size

        new_value = sum(test_positions.get(asset, 0) * prices.get(asset, 0) for asset in test_positions)

        results = {
            'var_limit_ok': True,
            'drawdown_limit_ok': True,
            'diversification_ok': True,
            'correlation_limit_ok': True
        }

        # Check VaR limit
        portfolio_var = self.calculate_portfolio_var(test_positions, prices)
        if portfolio_var > self.risk_limits.portfolio_var_limit:
            results['var_limit_ok'] = False

        # Check diversification
        active_positions = [asset for asset, size in test_positions.items() if size > 0]
        current_active = [asset for asset, size in positions.items() if size > 0]
        
        # Allow building up to minimum diversification from empty portfolio
        if len(current_active) < self.risk_limits.min_diversification:
            # Allow adding positions until we reach min diversification
            results['diversification_ok'] = True
        elif len(active_positions) < self.risk_limits.min_diversification:
            # Once we have min diversification, enforce the requirement
            results['diversification_ok'] = False

        # Check correlation limits
        if len(active_positions) > 1:
            correlations_ok = self._check_correlation_limits(test_positions, active_positions)
            results['correlation_limit_ok'] = correlations_ok

        return results

    def _check_correlation_limits(self, positions: Dict[str, float], assets: List[str]) -> bool:
        """Check if any asset pair exceeds correlation limit"""
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                if asset1 in self.correlation_matrix and asset2 in self.correlation_matrix[asset1]:
                    corr = abs(self.correlation_matrix[asset1][asset2])
                    if corr > self.risk_limits.max_correlation:
                        return False
        return True

    def update_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]) -> None:
        """Update correlation matrix from recent price data"""
        try:
            # Calculate returns for correlation
            returns_data = {}
            for asset, df in price_data.items():
                if len(df) > 30:
                    returns = df['close'].pct_change().dropna()
                    returns_data[asset] = returns

            if len(returns_data) > 1:
                returns_df = pd.DataFrame(returns_data)
                corr_matrix = returns_df.corr()

                # Update correlation matrix
                for asset1 in corr_matrix.columns:
                    if asset1 not in self.correlation_matrix:
                        self.correlation_matrix[asset1] = {}
                    for asset2 in corr_matrix.columns:
                        if asset1 != asset2:
                            self.correlation_matrix[asset1][asset2] = corr_matrix.loc[asset1, asset2]

        except Exception as e:
            logger.warning(f"Correlation update failed: {e}")

    def get_dynamic_stop_loss(self,
                            entry_price: float,
                            volatility: float,
                            regime: str = 'trending') -> float:
        """Calculate dynamic stop loss based on volatility and regime"""
        base_stop = entry_price * 0.02  # 2% base stop

        # Adjust for volatility
        vol_adjusted_stop = base_stop * (1 + volatility * 2)

        # Adjust for regime
        regime_adj = self.get_regime_adjustment(regime)
        final_stop = vol_adjusted_stop * regime_adj.stop_loss_multiplier

        return min(final_stop, entry_price * 0.05)  # Cap at 5%

    def get_dynamic_take_profit(self,
                              entry_price: float,
                              volatility: float,
                              regime: str = 'trending') -> float:
        """Calculate dynamic take profit based on volatility and regime"""
        base_tp = entry_price * 0.05  # 5% base take profit

        # Adjust for volatility (higher vol = higher target)
        vol_adjusted_tp = base_tp * (1 + volatility)

        # Adjust for regime
        regime_adj = self.get_regime_adjustment(regime)
        final_tp = vol_adjusted_tp * regime_adj.take_profit_multiplier

        return max(final_tp, entry_price * 0.02)  # Minimum 2%
