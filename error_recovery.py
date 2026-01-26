#!/usr/bin/env python3
"""
Comprehensive Error Recovery System
Bulletproof error handling for all trading operations
"""

import functools
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import traceback
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of error types for appropriate handling"""
    PRECISION_ERROR = "precision_error"
    AUTH_ERROR = "auth_error"
    DATA_ERROR = "data_error"
    NETWORK_ERROR = "network_error"
    API_LIMIT_ERROR = "api_limit_error"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    SYMBOL_ERROR = "symbol_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RecoveryAction:
    """Defines how to recover from specific error types"""
    action_type: str
    parameters: Dict[str, Any]
    max_attempts: int = 3
    delay_seconds: float = 1.0

class ErrorClassifier:
    """Classifies errors to determine appropriate recovery strategy"""
    
    def __init__(self):
        self.error_patterns = {
            ErrorType.PRECISION_ERROR: [
                "precision", "decimal", "rounding", "invalid quantity",
                "lot size", "tick size", "step size"
            ],
            ErrorType.AUTH_ERROR: [
                "unauthorized", "authentication", "invalid key", "api key",
                "signature", "403", "401", "forbidden"
            ],
            ErrorType.DATA_ERROR: [
                "no data", "empty response", "invalid data", "missing field",
                "malformed", "parsing error", "json", "xml"
            ],
            ErrorType.NETWORK_ERROR: [
                "connection", "timeout", "network", "dns", "socket",
                "502", "503", "504", "gateway"
            ],
            ErrorType.API_LIMIT_ERROR: [
                "rate limit", "too many requests", "429", "quota",
                "limit exceeded", "throttle"
            ],
            ErrorType.INSUFFICIENT_FUNDS: [
                "insufficient", "balance", "margin", "buying power",
                "not enough", "funds"
            ],
            ErrorType.SYMBOL_ERROR: [
                "invalid symbol", "unknown symbol", "not found",
                "unsupported", "delisted"
            ]
        }
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Classify error to determine recovery strategy"""
        error_msg = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern in error_msg or pattern in error_type_name:
                    logger.debug(f"Classified error as {error_type.value}: {pattern}")
                    return error_type
        
        logger.debug(f"Unclassified error: {error_msg}")
        return ErrorType.UNKNOWN_ERROR

class RecoveryManager:
    """Manages recovery strategies for different error types"""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.recovery_strategies = self._initialize_strategies()
        self.api_keys = []
        self.current_key_index = 0
        self.data_providers = ['alpaca', 'yahoo', 'binance']
        self.current_provider_index = 0
        
    def _initialize_strategies(self) -> Dict[ErrorType, RecoveryAction]:
        """Initialize recovery strategies for each error type"""
        return {
            ErrorType.PRECISION_ERROR: RecoveryAction(
                action_type="adjust_precision",
                parameters={},
                max_attempts=3,
                delay_seconds=0.1
            ),
            ErrorType.AUTH_ERROR: RecoveryAction(
                action_type="rotate_api_key", 
                parameters={},
                max_attempts=5,
                delay_seconds=2.0
            ),
            ErrorType.DATA_ERROR: RecoveryAction(
                action_type="switch_data_provider",
                parameters={},
                max_attempts=3,
                delay_seconds=1.0
            ),
            ErrorType.NETWORK_ERROR: RecoveryAction(
                action_type="exponential_backoff",
                parameters={"base_delay": 1.0, "max_delay": 30.0},
                max_attempts=5,
                delay_seconds=1.0
            ),
            ErrorType.API_LIMIT_ERROR: RecoveryAction(
                action_type="rate_limit_backoff",
                parameters={"backoff_multiplier": 2.0},
                max_attempts=3,
                delay_seconds=60.0
            ),
            ErrorType.INSUFFICIENT_FUNDS: RecoveryAction(
                action_type="reduce_position_size",
                parameters={"reduction_factor": 0.5},
                max_attempts=3,
                delay_seconds=0.1
            ),
            ErrorType.SYMBOL_ERROR: RecoveryAction(
                action_type="validate_symbol",
                parameters={},
                max_attempts=2,
                delay_seconds=0.1
            ),
            ErrorType.UNKNOWN_ERROR: RecoveryAction(
                action_type="generic_retry",
                parameters={},
                max_attempts=2,
                delay_seconds=2.0
            )
        }
    
    def execute_recovery(self, error: Exception, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute appropriate recovery action for the error"""
        error_type = self.classifier.classify_error(error)
        recovery_action = self.recovery_strategies.get(error_type)
        
        if not recovery_action:
            logger.error(f"No recovery strategy for error type: {error_type}")
            return {"success": False, "action": "no_strategy"}
        
        try:
            result = self._execute_action(recovery_action, error, operation_context)
            logger.info(f"Recovery action {recovery_action.action_type} executed: {result}")
            return result
            
        except Exception as recovery_error:
            logger.error(f"Recovery action failed: {recovery_error}")
            return {"success": False, "action": recovery_action.action_type, "error": str(recovery_error)}
    
    def _execute_action(self, action: RecoveryAction, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific recovery action"""
        action_map = {
            "adjust_precision": self._adjust_precision,
            "rotate_api_key": self._rotate_api_key,
            "switch_data_provider": self._switch_data_provider,
            "exponential_backoff": self._exponential_backoff,
            "rate_limit_backoff": self._rate_limit_backoff,
            "reduce_position_size": self._reduce_position_size,
            "validate_symbol": self._validate_symbol,
            "generic_retry": self._generic_retry
        }
        
        action_func = action_map.get(action.action_type)
        if not action_func:
            raise ValueError(f"Unknown recovery action: {action.action_type}")
        
        return action_func(error, context, action.parameters)
    
    def _adjust_precision(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust precision for price/quantity values"""
        from precision_manager import precision_manager
        
        symbol = context.get('symbol', '')
        price = context.get('price', 0)
        quantity = context.get('quantity', 0)
        
        if not symbol:
            return {"success": False, "reason": "no_symbol"}
        
        # Adjust precision using the precision manager
        adjusted_values = precision_manager.validate_precision(symbol, price, quantity)
        
        return {
            "success": True,
            "action": "precision_adjusted",
            "original_price": price,
            "adjusted_price": adjusted_values['price'],
            "original_quantity": quantity,
            "adjusted_quantity": adjusted_values['quantity']
        }
    
    def _rotate_api_key(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Rotate to next available API key"""
        if not self.api_keys:
            return {"success": False, "reason": "no_api_keys"}
        
        # Move to next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_index]
        
        return {
            "success": True,
            "action": "api_key_rotated",
            "new_key_index": self.current_key_index
        }
    
    def _switch_data_provider(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to next available data provider"""
        if not self.data_providers:
            return {"success": False, "reason": "no_providers"}
        
        # Move to next provider
        self.current_provider_index = (self.current_provider_index + 1) % len(self.data_providers)
        new_provider = self.data_providers[self.current_provider_index]
        
        return {
            "success": True,
            "action": "data_provider_switched", 
            "new_provider": new_provider
        }
    
    def _exponential_backoff(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Implement exponential backoff delay"""
        base_delay = params.get('base_delay', 1.0)
        max_delay = params.get('max_delay', 30.0)
        attempt = context.get('attempt', 1)
        
        delay = min(base_delay * (2 ** attempt), max_delay)
        time.sleep(delay)
        
        return {
            "success": True,
            "action": "exponential_backoff",
            "delay": delay,
            "attempt": attempt
        }
    
    def _rate_limit_backoff(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rate limit with longer backoff"""
        backoff_multiplier = params.get('backoff_multiplier', 2.0)
        base_delay = 60.0  # Start with 1 minute
        attempt = context.get('attempt', 1)
        
        delay = base_delay * (backoff_multiplier ** (attempt - 1))
        time.sleep(delay)
        
        return {
            "success": True,
            "action": "rate_limit_backoff",
            "delay": delay
        }
    
    def _reduce_position_size(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce position size for insufficient funds"""
        reduction_factor = params.get('reduction_factor', 0.5)
        original_quantity = context.get('quantity', 0)
        
        new_quantity = original_quantity * reduction_factor
        
        return {
            "success": True,
            "action": "position_size_reduced",
            "original_quantity": original_quantity,
            "new_quantity": new_quantity,
            "reduction_factor": reduction_factor
        }
    
    def _validate_symbol(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and potentially correct symbol format"""
        symbol = context.get('symbol', '')
        
        # Simple symbol validation/correction
        if '/' not in symbol and 'USD' in symbol:
            # Convert BTCUSD to BTC/USD
            base = symbol.replace('USD', '')
            corrected_symbol = f"{base}/USD"
            
            return {
                "success": True,
                "action": "symbol_corrected",
                "original_symbol": symbol,
                "corrected_symbol": corrected_symbol
            }
        
        return {
            "success": False,
            "action": "symbol_validation_failed",
            "symbol": symbol
        }
    
    def _generic_retry(self, error: Exception, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Generic retry with delay"""
        delay = 2.0
        time.sleep(delay)
        
        return {
            "success": True,
            "action": "generic_retry",
            "delay": delay
        }

class TradingOperationWithRecovery:
    """
    Wrapper for trading operations with automatic error recovery
    """
    
    def __init__(self):
        self.recovery_manager = RecoveryManager()
        self.max_recovery_attempts = 3
        
    def execute_with_recovery(self, operation: Callable, operation_context: Dict[str, Any], 
                            max_attempts: Optional[int] = None) -> Any:
        """
        Execute operation with automatic error recovery
        
        Args:
            operation: Function to execute
            operation_context: Context information for recovery (symbol, price, quantity, etc.)
            max_attempts: Maximum number of recovery attempts
            
        Returns:
            Result of successful operation
            
        Raises:
            MaxRetriesExceeded: If all recovery attempts fail
        """
        max_attempts = max_attempts or self.max_recovery_attempts
        last_error = None
        
        for attempt in range(max_attempts + 1):  # +1 for initial attempt
            try:
                # Update attempt number in context
                operation_context['attempt'] = attempt
                
                # Execute the operation
                result = operation()
                
                if attempt > 0:
                    logger.info(f"Operation succeeded after {attempt} recovery attempts")
                
                return result
                
            except Exception as error:
                last_error = error
                
                if attempt >= max_attempts:
                    logger.error(f"All recovery attempts exhausted for operation")
                    break
                
                logger.warning(f"Operation failed (attempt {attempt + 1}): {error}")
                
                # Attempt recovery
                recovery_result = self.recovery_manager.execute_recovery(error, operation_context)
                
                if not recovery_result.get('success', False):
                    logger.error(f"Recovery failed: {recovery_result}")
                    continue
                
                # Update context with recovery results
                operation_context.update(recovery_result)
                
                # Wait before retry if specified
                recovery_action = self.recovery_manager.recovery_strategies.get(
                    self.recovery_manager.classifier.classify_error(error)
                )
                if recovery_action and recovery_action.delay_seconds > 0:
                    time.sleep(recovery_action.delay_seconds)
        
        # All attempts failed
        raise MaxRetriesExceeded(f"Operation failed after {max_attempts} recovery attempts. Last error: {last_error}")

class MaxRetriesExceeded(Exception):
    """Raised when all recovery attempts are exhausted"""
    pass

# Decorator for automatic error recovery
def with_recovery(max_attempts: int = 3, context_keys: List[str] = None):
    """
    Decorator to add automatic error recovery to any function
    
    Args:
        max_attempts: Maximum recovery attempts
        context_keys: List of parameter names to include in recovery context
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build operation context from function parameters
            context = {}
            if context_keys:
                # Map positional args to context keys
                for i, key in enumerate(context_keys):
                    if i < len(args):
                        context[key] = args[i]
                
                # Add keyword args
                for key in context_keys:
                    if key in kwargs:
                        context[key] = kwargs[key]
            
            # Create operation wrapper
            def operation():
                return func(*args, **kwargs)
            
            # Execute with recovery
            recovery_handler = TradingOperationWithRecovery()
            return recovery_handler.execute_with_recovery(operation, context, max_attempts)
        
        return wrapper
    return decorator

# Global recovery instance
recovery_system = TradingOperationWithRecovery()

if __name__ == "__main__":
    # Test the error recovery system
    print("🔧 Testing Error Recovery System")
    print("=" * 40)
    
    # Test error classification
    classifier = ErrorClassifier()
    
    test_errors = [
        ValueError("Invalid quantity precision"),
        ConnectionError("Network timeout"),
        Exception("Unauthorized: invalid API key"),
        RuntimeError("Rate limit exceeded"),
        ValueError("Insufficient buying power")
    ]
    
    for error in test_errors:
        error_type = classifier.classify_error(error)
        print(f"Error: {error} -> {error_type.value}")
    
    print("\n✅ Error recovery system ready!")

# Global error recovery manager instance
error_recovery = RecoveryManager()

def execute_with_recovery(operation, *args, **kwargs):
    """Execute operation with error recovery"""
    return error_recovery.execute_recovery(operation, *args, **kwargs)

def get_circuit_breaker_status():
    """Get circuit breaker status"""
    return error_recovery.circuit_breaker.get_status()

def reset_circuit_breaker():
    """Reset circuit breaker"""
    error_recovery.circuit_breaker.reset()
