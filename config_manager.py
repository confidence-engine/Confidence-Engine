#!/usr/bin/env python3
"""
Unified Configuration Management System
Single source of truth for all system configuration
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ConfigSource(Enum):
    """Configuration source priority"""
    ENVIRONMENT = 1      # Highest priority
    RUNTIME = 2          # Runtime overrides
    CONFIG_FILE = 3      # Configuration files
    DEFAULTS = 4         # Default values (lowest priority)

@dataclass
class ConfigField:
    """Configuration field definition"""
    name: str
    default_value: Any
    description: str
    required: bool = False
    env_var: Optional[str] = None
    value_type: Type = str
    validator: Optional[callable] = None
    sensitive: bool = False  # Hide in logs

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    # Risk Management
    max_position_size: float = 1000.0
    max_daily_loss: float = 500.0
    position_size_pct: float = 0.02  # 2% of portfolio
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.04    # 4% take profit
    
    # Timeframes
    primary_timeframe: str = "15min"
    analysis_timeframe: str = "1h"
    signal_timeout_minutes: int = 60
    
    # Thresholds
    min_confidence: float = 0.6
    divergence_threshold: float = 0.3
    volume_threshold: float = 1.2
    
    # Safety
    paper_trading: bool = True
    max_daily_trades: int = 5
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"

@dataclass
class APIConfig:
    """API configuration"""
    # Alpaca
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # Perplexity
    pplx_api_keys: List[str] = field(default_factory=list)
    pplx_base_url: str = "https://api.perplexity.ai"
    pplx_model: str = "llama-3.1-sonar-small-128k-online"
    
    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Discord
    discord_webhook_url: str = ""

@dataclass
class DataConfig:
    """Data source configuration"""
    providers: List[str] = field(default_factory=lambda: ["alpaca", "yahoo"])
    primary_provider: str = "alpaca"
    cache_duration_hours: int = 1
    max_bars_per_request: int = 1000
    data_quality_checks: bool = True
    
    # Symbol configuration
    default_symbols: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    symbol_precision_override: Dict[str, Dict[str, int]] = field(default_factory=dict)

@dataclass
class MLConfig:
    """Machine Learning configuration"""
    model_path: str = "eval_runs/ml/latest/model.pt"
    feature_count: int = 37
    retrain_frequency_hours: int = 24
    
    # Model parameters
    ensemble_models: List[str] = field(default_factory=lambda: ["mlp", "lstm", "transformer", "cnn"])
    hidden_size: int = 128
    num_layers: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    
    # Training
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    health_check_interval: int = 30  # seconds
    log_level: str = "INFO"
    log_rotation_days: int = 7
    
    # Alerts
    critical_alert_enabled: bool = True
    warning_alert_enabled: bool = True
    alert_cooldown_minutes: int = 30
    
    # Metrics
    save_system_metrics: bool = True
    metrics_retention_days: int = 30
    performance_tracking: bool = True

@dataclass
class SystemConfig:
    """Complete system configuration"""
    # Configuration metadata
    config_version: str = "1.0.0"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: str = "development"
    
    # Component configurations
    trading: TradingConfig = field(default_factory=TradingConfig)
    apis: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

class ConfigManager:
    """
    Unified configuration management with environment override support
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration files
        self.main_config_file = self.config_dir / "system_config.yaml"
        self.runtime_config_file = self.config_dir / "runtime_overrides.json"
        self.secrets_file = self.config_dir / "secrets.env"
        
        # Configuration registry
        self.config_fields = self._register_config_fields()
        
        # Load configuration
        self._config = self._load_configuration()
    
    def _register_config_fields(self) -> Dict[str, ConfigField]:
        """Register all configuration fields with metadata"""
        fields = {
            # Trading Configuration
            "max_position_size": ConfigField(
                "max_position_size", 1000.0,
                "Maximum position size in USD",
                env_var="TB_MAX_POSITION_SIZE", value_type=float
            ),
            "paper_trading": ConfigField(
                "paper_trading", True,
                "Enable paper trading mode",
                env_var="TB_PAPER_TRADING", value_type=bool
            ),
            "stop_loss_pct": ConfigField(
                "stop_loss_pct", 0.02,
                "Stop loss percentage (0.02 = 2%)",
                env_var="TB_STOP_LOSS_PCT", value_type=float
            ),
            
            # API Configuration
            "alpaca_api_key": ConfigField(
                "alpaca_api_key", "",
                "Alpaca API key", required=False,  # Changed to False for testing
                env_var="ALPACA_API_KEY", sensitive=True
            ),
            "alpaca_secret_key": ConfigField(
                "alpaca_secret_key", "",
                "Alpaca secret key", required=False,  # Changed to False for testing
                env_var="ALPACA_SECRET_KEY", sensitive=True
            ),
            "telegram_bot_token": ConfigField(
                "telegram_bot_token", "",
                "Telegram bot token",
                env_var="TELEGRAM_BOT_TOKEN", sensitive=True
            ),
            
            # Perplexity API Keys (multiple)
            "pplx_api_key_1": ConfigField(
                "pplx_api_key_1", "",
                "Perplexity API key #1",
                env_var="PPLX_API_KEY_1", sensitive=True
            ),
            "pplx_api_key_2": ConfigField(
                "pplx_api_key_2", "",
                "Perplexity API key #2",
                env_var="PPLX_API_KEY_2", sensitive=True
            ),
            "pplx_api_key_3": ConfigField(
                "pplx_api_key_3", "",
                "Perplexity API key #3",
                env_var="PPLX_API_KEY_3", sensitive=True
            ),
            
            # Data Configuration
            "primary_provider": ConfigField(
                "primary_provider", "alpaca",
                "Primary data provider",
                env_var="TB_PRIMARY_PROVIDER"
            ),
            "cache_duration_hours": ConfigField(
                "cache_duration_hours", 1,
                "Data cache duration in hours",
                env_var="TB_CACHE_DURATION", value_type=int
            ),
            
            # ML Configuration
            "feature_count": ConfigField(
                "feature_count", 37,
                "Number of ML features",
                env_var="TB_FEATURE_COUNT", value_type=int
            ),
            "model_path": ConfigField(
                "model_path", "eval_runs/ml/latest/model.pt",
                "Path to ML model file",
                env_var="TB_MODEL_PATH"
            ),
            
            # Monitoring
            "log_level": ConfigField(
                "log_level", "INFO",
                "Logging level",
                env_var="TB_LOG_LEVEL",
                validator=lambda x: x.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
            "health_check_interval": ConfigField(
                "health_check_interval", 30,
                "Health check interval in seconds",
                env_var="TB_HEALTH_CHECK_INTERVAL", value_type=int
            )
        }
        
        return fields
    
    def _load_configuration(self) -> SystemConfig:
        """Load configuration from all sources with priority"""
        config = SystemConfig()
        
        # 1. Start with defaults (already set in dataclass)
        
        # 2. Load from YAML config file
        if self.main_config_file.exists():
            try:
                with open(self.main_config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                self._merge_config(config, file_config, ConfigSource.CONFIG_FILE)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # 3. Load runtime overrides
        if self.runtime_config_file.exists():
            try:
                with open(self.runtime_config_file, 'r') as f:
                    runtime_config = json.load(f)
                self._merge_config(config, runtime_config, ConfigSource.RUNTIME)
            except Exception as e:
                logger.warning(f"Failed to load runtime config: {e}")
        
        # 4. Override with environment variables (highest priority)
        self._load_from_environment(config)
        
        # 5. Validate configuration
        self._validate_config(config)
        
        return config
    
    def _merge_config(self, config: SystemConfig, source_config: Dict[str, Any], source: ConfigSource):
        """Merge configuration from source into config object"""
        def merge_nested(target, source, path=""):
            for key, value in source.items():
                if hasattr(target, key):
                    if isinstance(value, dict) and hasattr(getattr(target, key), '__dict__'):
                        # Recursively merge nested objects
                        merge_nested(getattr(target, key), value, f"{path}.{key}")
                    else:
                        # Set the value
                        setattr(target, key, value)
                        logger.debug(f"Config [{source.name}]: {path}.{key} = {value}")
        
        merge_nested(config, source_config)
    
    def _load_from_environment(self, config: SystemConfig):
        """Load configuration from environment variables"""
        for field_name, field_def in self.config_fields.items():
            if field_def.env_var and field_def.env_var in os.environ:
                env_value = os.environ[field_def.env_var]
                
                # Convert type
                try:
                    if field_def.value_type == bool:
                        typed_value = env_value.lower() in ['true', '1', 'yes', 'on']
                    elif field_def.value_type == int:
                        typed_value = int(env_value)
                    elif field_def.value_type == float:
                        typed_value = float(env_value)
                    else:
                        typed_value = env_value
                    
                    # Apply validator if present
                    if field_def.validator and not field_def.validator(typed_value):
                        logger.warning(f"Environment variable {field_def.env_var} failed validation")
                        continue
                    
                    # Set the value (need to navigate to correct nested object)
                    self._set_nested_value(config, field_name, typed_value)
                    
                    # Log (but hide sensitive values)
                    log_value = "***" if field_def.sensitive else typed_value
                    logger.debug(f"Config [ENV]: {field_name} = {log_value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert env var {field_def.env_var}: {e}")
        
        # Special handling for multiple Perplexity API keys
        pplx_keys = []
        for i in range(1, 10):  # Support up to 9 keys
            key_var = f"PPLX_API_KEY_{i}"
            if key_var in os.environ:
                pplx_keys.append(os.environ[key_var])
        
        if pplx_keys:
            config.apis.pplx_api_keys = pplx_keys
            logger.debug(f"Config [ENV]: Loaded {len(pplx_keys)} Perplexity API keys")
    
    def _set_nested_value(self, config: SystemConfig, field_name: str, value: Any):
        """Set nested configuration value"""
        # Map field names to nested paths
        field_mapping = {
            "max_position_size": ("trading", "max_position_size"),
            "paper_trading": ("trading", "paper_trading"),
            "stop_loss_pct": ("trading", "stop_loss_pct"),
            "alpaca_api_key": ("apis", "alpaca_api_key"),
            "alpaca_secret_key": ("apis", "alpaca_secret_key"),
            "telegram_bot_token": ("apis", "telegram_bot_token"),
            "primary_provider": ("data", "primary_provider"),
            "cache_duration_hours": ("data", "cache_duration_hours"),
            "feature_count": ("ml", "feature_count"),
            "model_path": ("ml", "model_path"),
            "log_level": ("monitoring", "log_level"),
            "health_check_interval": ("monitoring", "health_check_interval")
        }
        
        if field_name in field_mapping:
            section, attr = field_mapping[field_name]
            setattr(getattr(config, section), attr, value)
    
    def _validate_config(self, config: SystemConfig):
        """Validate configuration values"""
        errors = []
        
        # Check required fields
        for field_name, field_def in self.config_fields.items():
            if field_def.required:
                value = self._get_nested_value(config, field_name)
                if not value:
                    errors.append(f"Required field {field_name} is empty")
        
        # Trading validations
        if config.trading.max_position_size <= 0:
            errors.append("max_position_size must be positive")
        
        if not 0 < config.trading.position_size_pct < 1:
            errors.append("position_size_pct must be between 0 and 1")
        
        # API validations
        if config.trading.paper_trading and not config.apis.alpaca_api_key:
            logger.warning("Paper trading enabled but no Alpaca API key configured")
        
        # ML validations
        if config.ml.feature_count <= 0:
            errors.append("feature_count must be positive")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _get_nested_value(self, config: SystemConfig, field_name: str) -> Any:
        """Get nested configuration value"""
        field_mapping = {
            "max_position_size": ("trading", "max_position_size"),
            "paper_trading": ("trading", "paper_trading"),
            "alpaca_api_key": ("apis", "alpaca_api_key"),
            # ... (same mapping as _set_nested_value)
        }
        
        if field_name in field_mapping:
            section, attr = field_mapping[field_name]
            return getattr(getattr(config, section), attr)
        return None
    
    def get_config(self) -> SystemConfig:
        """Get current system configuration"""
        return self._config
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        config_file = config_file or self.main_config_file
        
        # Convert to dictionary (excluding sensitive fields)
        config_dict = self._to_dict(exclude_sensitive=True)
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def set_runtime_override(self, key: str, value: Any):
        """Set runtime configuration override"""
        try:
            # Load existing runtime config
            runtime_config = {}
            if self.runtime_config_file.exists():
                with open(self.runtime_config_file, 'r') as f:
                    runtime_config = json.load(f)
            
            # Set the override
            runtime_config[key] = value
            
            # Save back to file
            with open(self.runtime_config_file, 'w') as f:
                json.dump(runtime_config, f, indent=2)
            
            # Reload configuration
            self._config = self._load_configuration()
            
            logger.info(f"Runtime override set: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to set runtime override: {e}")
            raise
    
    def clear_runtime_overrides(self):
        """Clear all runtime configuration overrides"""
        if self.runtime_config_file.exists():
            self.runtime_config_file.unlink()
        
        # Reload configuration
        self._config = self._load_configuration()
        logger.info("Runtime overrides cleared")
    
    def _to_dict(self, exclude_sensitive: bool = True) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = asdict(self._config)
        
        if exclude_sensitive:
            # Remove sensitive fields
            sensitive_fields = [
                ["apis", "alpaca_api_key"],
                ["apis", "alpaca_secret_key"],
                ["apis", "pplx_api_keys"],
                ["apis", "telegram_bot_token"],
                ["apis", "discord_webhook_url"]
            ]
            
            for field_path in sensitive_fields:
                current = config_dict
                for part in field_path[:-1]:
                    if part in current:
                        current = current[part]
                    else:
                        break
                else:
                    if field_path[-1] in current:
                        current[field_path[-1]] = "***"
        
        return config_dict
    
    def get_connection_string(self, service: str) -> str:
        """Get connection string for external services"""
        config = self._config
        
        if service == "alpaca":
            return f"key={config.apis.alpaca_api_key[:8]}...&base_url={config.apis.alpaca_base_url}"
        elif service == "telegram":
            return f"bot_token={config.apis.telegram_bot_token[:8]}...&chat_id={config.apis.telegram_chat_id}"
        elif service == "perplexity":
            key_count = len(config.apis.pplx_api_keys)
            return f"keys={key_count}&model={config.apis.pplx_model}"
        else:
            return f"Unknown service: {service}"
    
    def validate_api_credentials(self) -> Dict[str, bool]:
        """Validate all API credentials"""
        results = {}
        config = self._config
        
        # Alpaca
        results["alpaca"] = bool(config.apis.alpaca_api_key and config.apis.alpaca_secret_key)
        
        # Telegram
        results["telegram"] = bool(config.apis.telegram_bot_token)
        
        # Perplexity
        results["perplexity"] = bool(config.apis.pplx_api_keys)
        
        # Discord
        results["discord"] = bool(config.apis.discord_webhook_url)
        
        return results

# Global configuration manager
config_manager = ConfigManager()

def get_config() -> SystemConfig:
    """Get current system configuration"""
    return config_manager.get_config()

def get_trading_config() -> TradingConfig:
    """Get trading configuration"""
    return config_manager.get_config().trading

def get_api_config() -> APIConfig:
    """Get API configuration"""
    return config_manager.get_config().apis

def get_data_config() -> DataConfig:
    """Get data configuration"""
    return config_manager.get_config().data

def get_ml_config() -> MLConfig:
    """Get ML configuration"""
    return config_manager.get_config().ml

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config_manager.get_config().monitoring

def set_runtime_override(key: str, value: Any):
    """Set runtime configuration override"""
    config_manager.set_runtime_override(key, value)

if __name__ == "__main__":
    # Test the configuration system
    print("⚙️  Testing Configuration Management System")
    print("=" * 50)
    
    config = get_config()
    
    print(f"\nEnvironment: {config.environment}")
    print(f"Config Version: {config.config_version}")
    print(f"Last Updated: {config.last_updated}")
    
    print("\nTrading Configuration:")
    print(f"  Paper Trading: {config.trading.paper_trading}")
    print(f"  Max Position Size: ${config.trading.max_position_size:,.2f}")
    print(f"  Stop Loss: {config.trading.stop_loss_pct:.1%}")
    print(f"  Take Profit: {config.trading.take_profit_pct:.1%}")
    
    print("\nData Configuration:")
    print(f"  Primary Provider: {config.data.primary_provider}")
    print(f"  Providers: {', '.join(config.data.providers)}")
    print(f"  Cache Duration: {config.data.cache_duration_hours}h")
    
    print("\nML Configuration:")
    print(f"  Feature Count: {config.ml.feature_count}")
    print(f"  Model Path: {config.ml.model_path}")
    print(f"  Ensemble Models: {', '.join(config.ml.ensemble_models)}")
    
    print("\nAPI Credentials Validation:")
    credentials = config_manager.validate_api_credentials()
    for service, valid in credentials.items():
        status = "✅" if valid else "❌"
        print(f"  {status} {service.title()}: {'Configured' if valid else 'Not configured'}")
    
    print("\n✅ Configuration system ready!")
