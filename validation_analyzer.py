"""
Validation analyzer for trade validation and reporting
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class ValidationAnalyzer:
    """Analyzes and validates trading decisions"""
    
    def __init__(self):
        self.validation_history = []
    
    def validate_trade_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a trading signal before execution.
        
        Args:
            signal_data: Dict containing signal information
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'is_valid': True,
            'confidence': 1.0,
            'warnings': [],
            'errors': [],
            'timestamp': datetime.now()
        }
        
        # Basic validations
        required_fields = ['symbol', 'direction', 'confidence']
        for field in required_fields:
            if field not in signal_data:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # Confidence validation
        if 'confidence' in signal_data:
            confidence = signal_data['confidence']
            if not 0 <= confidence <= 1:
                validation_result['warnings'].append(f"Confidence {confidence} outside [0,1] range")
                validation_result['confidence'] *= 0.8
        
        # Symbol validation
        if 'symbol' in signal_data:
            symbol = signal_data['symbol']
            if '/' not in symbol or 'USD' not in symbol:
                validation_result['warnings'].append(f"Unusual symbol format: {symbol}")
                validation_result['confidence'] *= 0.9
        
        # Direction validation
        if 'direction' in signal_data:
            direction = signal_data['direction']
            if direction not in ['long', 'short', 'buy', 'sell']:
                validation_result['errors'].append(f"Invalid direction: {direction}")
                validation_result['is_valid'] = False
        
        self.validation_history.append(validation_result)
        
        if validation_result['errors']:
            logger.error(f"Validation failed: {validation_result['errors']}")
        elif validation_result['warnings']:
            logger.warning(f"Validation warnings: {validation_result['warnings']}")
        else:
            logger.info("Signal validation passed")
        
        return validation_result
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate a validation report"""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'valid_signals': 0,
                'invalid_signals': 0,
                'average_confidence': 0.0,
                'common_warnings': [],
                'last_updated': datetime.now()
            }
        
        total = len(self.validation_history)
        valid = sum(1 for v in self.validation_history if v['is_valid'])
        invalid = total - valid
        
        # Calculate average confidence
        confidences = [v['confidence'] for v in self.validation_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Find common warnings
        all_warnings = []
        for v in self.validation_history:
            all_warnings.extend(v['warnings'])
        
        warning_counts = {}
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        common_warnings = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_validations': total,
            'valid_signals': valid,
            'invalid_signals': invalid,
            'success_rate': valid / total if total > 0 else 0.0,
            'average_confidence': avg_confidence,
            'common_warnings': common_warnings,
            'last_updated': datetime.now()
        }
    
    def clear_history(self, keep_last_n: int = 100):
        """Clear old validation history, keeping only the last N entries"""
        if len(self.validation_history) > keep_last_n:
            self.validation_history = self.validation_history[-keep_last_n:]
            logger.info(f"Cleared validation history, keeping last {keep_last_n} entries")
