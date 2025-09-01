import os
import subprocess
import sys
import time
import logging
import json
import functools
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
import git
import numpy as np
import pandas as pd
import httpx
import torch
from dotenv import load_dotenv

# Ensure project root on sys.path for local imports when executed directly
_THIS_DIR = Path(__file__).parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# Load environment variables from .env file
load_dotenv(_PROJ_ROOT / ".env")

try:
    from alpaca_trade_api.rest import REST  # type: ignore
except Exception:
    REST = None  # type: ignore

from config import settings
from scripts.discord_sender import send_discord_digest_to
from telegram_bot import send_message as send_telegram
from scripts.retry_utils import retry_call, RETRY_STATUS_CODES

# ========== VALIDATION IMPROVEMENTS ==========
try:
    from validation_analyzer import ValidationAnalyzer
    from paper_trading_optimizer import PaperTradingOptimizer
    VALIDATION_TOOLS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Validation tools imported successfully")
except ImportError as e:
    VALIDATION_TOOLS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Validation tools not available: {e}")

# ========== ZERO-COST ENHANCEMENTS ==========

import asyncio
import concurrent.futures

class CircuitBreaker:
    """Prevents API failures from crashing the agent"""
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker HALF_OPEN for {func.__name__}")
                else:
                    logger.warning(f"Circuit breaker OPEN for {func.__name__}")
                    return None  # Return None instead of raising exception
                    
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker CLOSED for {func.__name__}")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker OPENED for {func.__name__} after {self.failure_count} failures")
                
                logger.error(f"Circuit breaker caught error in {func.__name__}: {e}")
                return None  # Return None instead of raising
        return wrapper

class PerformanceTracker:
    """Track agent performance and errors"""
    def __init__(self):
        self.start_time = time.time()
        self.trade_count = 0
        self.error_count = 0
        self.last_heartbeat = time.time()
        
    def heartbeat(self):
        self.last_heartbeat = time.time()
        
    def record_trade(self):
        self.trade_count += 1
        
    def record_error(self):
        self.error_count += 1
        
    def get_stats(self):
        uptime = time.time() - self.start_time
        return {
            "uptime_hours": uptime / 3600,
            "trade_count": self.trade_count,
            "error_count": self.error_count,
            "last_heartbeat_ago": time.time() - self.last_heartbeat
        }

class TradingDatabase:
    """SQLite database for better data management"""
    def __init__(self, db_path="enhanced_trading.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    confidence REAL,
                    pnl REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trades_today INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    uptime_hours REAL,
                    error_count INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    position_side TEXT,
                    position_qty REAL DEFAULT 0,
                    position_entry_price REAL DEFAULT 0,
                    last_entry_time REAL,
                    order_id TEXT,
                    pnl_today REAL DEFAULT 0,
                    pnl_total REAL DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    UNIQUE(symbol)
                )
            """)
            
    def log_trade(self, run_id: str, symbol: str, action: str, price: float, 
                  confidence: float = None, pnl: float = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (timestamp, run_id, symbol, action, price, confidence, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                run_id,
                symbol,
                action,
                price,
                confidence,
                pnl
            ))
            
    def get_recent_performance(self, hours: int = 24):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as trades,
                       AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                       SUM(pnl) as total_pnl
                FROM trades 
                WHERE timestamp > datetime('now', '-{} hours')
                AND pnl IS NOT NULL
            """.format(hours))
            row = cursor.fetchone()
            return {
                "trades": row[0] or 0,
                "win_rate": row[1] or 0,
                "total_pnl": row[2] or 0
            }
            
    def save_position_state(self, symbol: str, state: dict):
        """Save position state to SQLite instead of JSON"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions 
                (symbol, position_side, position_qty, position_entry_price, 
                 last_entry_time, order_id, pnl_today, pnl_total, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                state.get("position_side"),
                state.get("position_qty", 0),
                state.get("position_entry_price", 0),
                state.get("last_entry_time"),
                state.get("order_id"),
                state.get("pnl_today", 0),
                state.get("pnl_total", 0),
                datetime.utcnow().isoformat()
            ))
            
    def load_position_state(self, symbol: str) -> dict:
        """Load position state from SQLite with JSON fallback"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT position_side, position_qty, position_entry_price, 
                       last_entry_time, order_id, pnl_today, pnl_total
                FROM positions WHERE symbol = ?
            """, (symbol,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "position_side": row[0],
                    "position_qty": row[1] or 0,
                    "position_entry_price": row[2] or 0,
                    "last_entry_time": row[3],
                    "order_id": row[4],
                    "pnl_today": row[5] or 0,
                    "pnl_total": row[6] or 0
                }
            else:
                # Fallback to JSON file and migrate
                try:
                    json_state = load_state(symbol)
                    if json_state:
                        self.save_position_state(symbol, json_state)
                        logger.info(f"Migrated {symbol} state from JSON to SQLite")
                        return json_state
                except Exception as e:
                    logger.warning(f"Could not load JSON state for {symbol}: {e}")
                    
                # Return default empty state
                return {
                    "position_side": None,
                    "position_qty": 0,
                    "position_entry_price": 0,
                    "last_entry_time": None,
                    "order_id": None,
                    "pnl_today": 0,
                    "pnl_total": 0
                }

# Initialize zero-cost enhancements
performance_tracker = PerformanceTracker()
trading_db = TradingDatabase()

# ========== ASYNC PROCESSING ENHANCEMENTS ==========

class AsyncProcessor:
    """Async processing for parallel data fetching"""
    def __init__(self, max_workers=3):
        self.max_workers = max_workers
        
    def run_parallel(self, tasks):
        """Run multiple tasks in parallel using thread pool"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(task['func'], *task.get('args', []), **task.get('kwargs', {})): task for task in tasks}
            results = {}
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                task_name = task['name']
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    results[task_name] = None
                    performance_tracker.record_error()
                    
            return results
            
    def fetch_all_symbol_data(self, symbol):
        """Fetch all data for a symbol in parallel"""
        tasks = [
            {
                'name': 'bars_15m',
                'func': fetch_bars,
                'args': [symbol, TF_FAST],
                'kwargs': {'lookback': 200}
            },
            {
                'name': 'bars_1h', 
                'func': fetch_bars,
                'args': [symbol, TF_SLOW],
                'kwargs': {'lookback': 200}
            }
        ]
        
        # Add sentiment fetch if not offline
        if not OFFLINE:
            tasks.append({
                'name': 'sentiment',
                'func': self.fetch_sentiment_for_symbol,
                'args': [symbol]
            })
            
        return self.run_parallel(tasks)
        
    def fetch_sentiment_for_symbol(self, symbol):
        """Fetch sentiment for a specific symbol"""
        try:
            if OFFLINE:
                return 0.6  # Default sentiment
                
            api_news = _rest()
            if api_news:
                news_resp = api_news.get_news(_normalize_symbol(symbol), limit=10)
                headlines = [item.headline for item in news_resp]
                sentiment, _ = sentiment_via_perplexity(headlines)
                return sentiment
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"Error fetching sentiment for {symbol}: {e}")
            return 0.5

# Initialize async processor
async_processor = AsyncProcessor(max_workers=3)

# ========== HEALTH MONITORING SYSTEM ==========

class HealthMonitor:
    """Comprehensive health monitoring for the trading system"""
    def __init__(self):
        self.checks = {}
        self.alerts_sent = set()
        self.last_health_check = time.time()
        
    def register_check(self, name: str, check_func, warning_threshold=None, critical_threshold=None):
        """Register a health check"""
        self.checks[name] = {
            'function': check_func,
            'warning_threshold': warning_threshold,
            'critical_threshold': critical_threshold,
            'last_result': None,
            'last_check': None
        }
        
    def run_health_checks(self):
        """Run all registered health checks"""
        results = {}
        current_time = time.time()
        
        for name, check in self.checks.items():
            try:
                result = check['function']()
                check['last_result'] = result
                check['last_check'] = current_time
                
                # Determine status
                status = 'healthy'
                if check['critical_threshold'] is not None and result > check['critical_threshold']:
                    status = 'critical'
                elif check['warning_threshold'] is not None and result > check['warning_threshold']:
                    status = 'warning'
                    
                results[name] = {
                    'value': result,
                    'status': status,
                    'timestamp': current_time
                }
                
                # Send alerts for critical issues
                if status == 'critical' and name not in self.alerts_sent:
                    self.send_health_alert(name, result, status)
                    self.alerts_sent.add(name)
                elif status == 'healthy' and name in self.alerts_sent:
                    # Clear alert once healthy
                    self.alerts_sent.discard(name)
                    
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = {
                    'value': None,
                    'status': 'error',
                    'error': str(e),
                    'timestamp': current_time
                }
                
        self.last_health_check = current_time
        return results
        
    def send_health_alert(self, check_name: str, value, status: str):
        """Send health alert via notifications"""
        try:
            message = f"🚨 HEALTH ALERT: {check_name}\nStatus: {status}\nValue: {value}\nTime: {datetime.now().strftime('%H:%M:%S')}"
            
            # Try to send via existing notification channels
            if hasattr(send_telegram, '__call__'):
                try:
                    send_telegram(message)
                except Exception:
                    pass
                    
            logger.error(f"HEALTH ALERT: {check_name} = {value} ({status})")
        except Exception as e:
            logger.error(f"Failed to send health alert: {e}")
            
    def get_system_health_summary(self):
        """Get overall system health summary"""
        if not self.checks:
            return {'status': 'no_checks', 'message': 'No health checks configured'}
            
        latest_results = {}
        critical_count = 0
        warning_count = 0
        healthy_count = 0
        error_count = 0
        
        for name, check in self.checks.items():
            if check['last_result'] is not None:
                # Re-evaluate status
                result = check['last_result']
                status = 'healthy'
                if check['critical_threshold'] is not None and result > check['critical_threshold']:
                    status = 'critical'
                    critical_count += 1
                elif check['warning_threshold'] is not None and result > check['warning_threshold']:
                    status = 'warning'
                    warning_count += 1
                else:
                    healthy_count += 1
                    
                latest_results[name] = {'value': result, 'status': status}
            else:
                error_count += 1
                latest_results[name] = {'value': None, 'status': 'error'}
        
        # Determine overall status
        if critical_count > 0:
            overall_status = 'critical'
        elif warning_count > 0:
            overall_status = 'warning'
        elif error_count > 0:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
            
        return {
            'status': overall_status,
            'checks': latest_results,
            'summary': {
                'critical': critical_count,
                'warning': warning_count,
                'healthy': healthy_count,
                'error': error_count
            },
            'last_check': self.last_health_check
        }

# Initialize health monitor and register checks
health_monitor = HealthMonitor()

# Register system health checks
def check_database_connection():
    """Check if SQLite database is accessible"""
    try:
        with sqlite3.connect(trading_db.db_path) as conn:
            conn.execute("SELECT 1").fetchone()
        return 0  # Healthy
    except Exception:
        return 1  # Critical

def check_performance_tracker():
    """Check performance tracker responsiveness"""
    try:
        stats = performance_tracker.get_stats()
        error_rate = stats['error_count'] / max(1, stats['trade_count'] + stats['error_count'])
        return error_rate * 100  # Return error rate as percentage
    except Exception:
        return 100  # Critical

def check_memory_usage():
    """Check actual memory usage in MB"""
    import psutil
    import os
    try:
        # Get current process memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
        return memory_mb
    except Exception:
        # Fallback to object count if psutil not available
        import gc
        objects = len(gc.get_objects())
        # Rough estimate: assume ~1KB per object on average
        estimated_mb = objects / 1000  # Convert to MB
        return estimated_mb

def check_api_connectivity():
    """Check if we can connect to Alpaca API"""
    try:
        if OFFLINE:
            return 0  # Skip in offline mode
        api = _rest()
        if api:
            account = api.get_account()
            return 0 if account else 1
        return 1
    except Exception:
        return 1

# Register all health checks
health_monitor.register_check('database', check_database_connection, critical_threshold=0.5)
health_monitor.register_check('performance', check_performance_tracker, warning_threshold=10, critical_threshold=25)
health_monitor.register_check('memory', check_memory_usage, warning_threshold=1024, critical_threshold=2048)
health_monitor.register_check('api_connectivity', check_api_connectivity, critical_threshold=0.5)

# ========== CONFIGURATION MANAGEMENT SYSTEM ==========

class ConfigManager:
    """Advanced configuration management with validation and hot reloading"""
    def __init__(self):
        self.config_cache = {}
        self.validators = {}
        self.change_callbacks = {}
        self.last_reload = time.time()
        
    def register_validator(self, key: str, validator_func):
        """Register a validator function for a config key"""
        self.validators[key] = validator_func
        
    def register_change_callback(self, key: str, callback_func):
        """Register a callback for when a config value changes"""
        self.change_callbacks[key] = callback_func
        
    def get_config(self, key: str, default=None, reload_if_stale=True):
        """Get configuration value with optional validation and hot reload"""
        if reload_if_stale and time.time() - self.last_reload > 60:  # Reload every minute
            self.reload_config()
            
        # Try environment variable first
        env_value = os.getenv(key, default)
        
        # Apply validator if registered
        if key in self.validators:
            try:
                env_value = self.validators[key](env_value)
            except Exception as e:
                logger.warning(f"Config validation failed for {key}: {e}, using default")
                env_value = default
                
        # Check if value changed and call callback
        if key in self.config_cache and self.config_cache[key] != env_value:
            if key in self.change_callbacks:
                try:
                    self.change_callbacks[key](self.config_cache[key], env_value)
                except Exception as e:
                    logger.error(f"Config change callback failed for {key}: {e}")
                    
        self.config_cache[key] = env_value
        return env_value
        
    def reload_config(self):
        """Force reload configuration from environment"""
        logger.debug("Reloading configuration from environment")
        self.last_reload = time.time()
        
        # Reload .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except Exception as e:
            logger.debug(f"Could not reload .env file: {e}")
            
    def get_config_summary(self):
        """Get summary of all configuration values"""
        summary = {}
        
        # Key configuration values to monitor
        important_keys = [
            'TB_NO_TRADE', 'TB_OFFLINE', 'TB_MULTI_ASSET', 'TB_USE_ENHANCED_RISK',
            'TB_AUTOCOMMIT_PUSH', 'TB_TRADER_NOTIFY_HEARTBEAT', 'TB_HEARTBEAT_EVERY_N',
            'TB_RETRY_ATTEMPTS', 'SYMBOL', 'TF_FAST', 'TF_SLOW'
        ]
        
        for key in important_keys:
            summary[key] = os.getenv(key, 'NOT_SET')
            
        return summary

# Configuration validators
def validate_boolean(value):
    """Validate boolean configuration values"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('1', 'true', 'yes', 'on')
    return bool(value)

def validate_positive_int(value):
    """Validate positive integer configuration values"""
    if value is None:
        return None
    val = int(value)
    if val < 0:
        raise ValueError(f"Value must be positive, got {val}")
    return val

def validate_timeframe(value):
    """Validate timeframe strings"""
    if value is None:
        return None
    valid_timeframes = ['1min', '5min', '15min', '1hour', '1day']
    if value not in valid_timeframes:
        raise ValueError(f"Invalid timeframe {value}, must be one of {valid_timeframes}")
    return value

# Initialize configuration manager
config_manager = ConfigManager()

# Register validators
config_manager.register_validator('TB_NO_TRADE', validate_boolean)
config_manager.register_validator('TB_OFFLINE', validate_boolean)
config_manager.register_validator('TB_MULTI_ASSET', validate_boolean)
config_manager.register_validator('TB_USE_ENHANCED_RISK', validate_boolean)
config_manager.register_validator('TB_HEARTBEAT_EVERY_N', validate_positive_int)
config_manager.register_validator('TB_RETRY_ATTEMPTS', validate_positive_int)
config_manager.register_validator('TF_FAST', validate_timeframe)
config_manager.register_validator('TF_SLOW', validate_timeframe)

# Register change callbacks
def on_no_trade_change(old_value, new_value):
    logger.info(f"Trading mode changed: NO_TRADE {old_value} -> {new_value}")

def on_offline_change(old_value, new_value):
    logger.info(f"Offline mode changed: OFFLINE {old_value} -> {new_value}")

config_manager.register_change_callback('TB_NO_TRADE', on_no_trade_change)
config_manager.register_change_callback('TB_OFFLINE', on_offline_change)

# ========== ADVANCED ASYNC FEATURES ==========

import queue
import threading
from typing import Callable, Any

class AsyncTaskQueue:
    """Advanced async task queue for background processing"""
    def __init__(self, max_workers=2):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.max_workers = max_workers
        self.shutdown_event = threading.Event()
        self.start_workers()
        
    def start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, name=f"AsyncWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
    def _worker(self):
        """Worker thread function"""
        while not self.shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                    
                task_id, func, args, kwargs = task
                try:
                    result = func(*args, **kwargs)
                    self.result_queue.put((task_id, 'success', result))
                except Exception as e:
                    self.result_queue.put((task_id, 'error', str(e)))
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs):
        """Submit a task for background processing"""
        self.task_queue.put((task_id, func, args, kwargs))
        
    def get_results(self, timeout=0.1):
        """Get completed task results"""
        results = {}
        try:
            while True:
                task_id, status, result = self.result_queue.get(timeout=timeout)
                results[task_id] = {'status': status, 'result': result}
        except queue.Empty:
            pass
        return results
        
    def shutdown(self):
        """Shutdown the task queue"""
        self.shutdown_event.set()
        # Signal workers to stop
        for _ in self.workers:
            self.task_queue.put(None)

class CacheManager:
    """Simple cache manager for expensive operations"""
    def __init__(self, default_ttl=300):  # 5 minutes default TTL
        self.cache = {}
        self.default_ttl = default_ttl
        
    def get(self, key: str):
        """Get cached value if not expired"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any, ttl: int = None):
        """Set cached value with TTL"""
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
        
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = [k for k, (_, expiry) in self.cache.items() if current_time >= expiry]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)

class AsyncEnhancedProcessor(AsyncProcessor):
    """Enhanced async processor with queuing and caching"""
    def __init__(self, max_workers=3):
        super().__init__(max_workers)
        self.task_queue = AsyncTaskQueue(max_workers=2)
        self.cache = CacheManager(default_ttl=180)  # 3 minute cache
        
    def fetch_cached_sentiment(self, symbol: str):
        """Fetch sentiment with caching"""
        cache_key = f"sentiment_{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Using cached sentiment for {symbol}")
            return cached
            
        # Fetch fresh sentiment
        sentiment = self.fetch_sentiment_for_symbol(symbol)
        if sentiment is not None:
            self.cache.set(cache_key, sentiment, ttl=120)  # 2 minute cache
        return sentiment
        
    def fetch_cached_bars(self, symbol: str, timeframe: str, lookback: int):
        """Fetch bars with caching"""
        cache_key = f"bars_{symbol}_{timeframe}_{lookback}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Using cached bars for {symbol} {timeframe}")
            return cached
            
        # Fetch fresh bars
        bars = fetch_bars(symbol, timeframe, lookback=lookback)
        if bars is not None and len(bars) > 0:
            self.cache.set(cache_key, bars, ttl=60)  # 1 minute cache for bars
        return bars
        
    def submit_background_analysis(self, symbol: str, bars_15m, bars_1h):
        """Submit technical analysis for background processing"""
        task_id = f"analysis_{symbol}_{int(time.time())}"
        self.task_queue.submit_task(task_id, self._background_analysis, symbol, bars_15m, bars_1h)
        return task_id
        
    def _background_analysis(self, symbol: str, bars_15m, bars_1h):
        """Perform technical analysis in background"""
        try:
            # Calculate additional indicators in background
            indicators = {
                'rsi_14': self._calculate_rsi(bars_15m['close'], 14),
                'bollinger_bands': self._calculate_bollinger_bands(bars_15m['close']),
                'volume_sma': bars_15m['volume'].rolling(window=20).mean().iloc[-1],
                'price_change_1h': (bars_1h['close'].iloc[-1] - bars_1h['close'].iloc[-2]) / bars_1h['close'].iloc[-2] * 100
            }
            return indicators
        except Exception as e:
            logger.error(f"Background analysis failed for {symbol}: {e}")
            return {}
            
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if len(rsi) > 0 else 50
        except Exception:
            return 50
            
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            current_price = prices.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            
            # Return position within bands (0 = at lower band, 1 = at upper band)
            position = (current_price - current_lower) / (current_upper - current_lower)
            return max(0, min(1, position))
        except Exception:
            return 0.5
            
    def cleanup(self):
        """Cleanup resources"""
        self.cache.clear_expired()
        self.task_queue.shutdown()

# Initialize enhanced async processor
enhanced_async_processor = AsyncEnhancedProcessor(max_workers=3)

# ========== END ADVANCED ASYNC FEATURES ==========

# Import enhanced components
try:
    # Import from the root-level advanced_risk_manager.py (not scripts/)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Add project root to path
    from advanced_risk_manager import AdvancedRiskManager, KellyPositionSizer
    from market_regime_detector import MarketRegimeDetector
    from ensemble_ml_models import TradingEnsemble
    from adaptive_strategy import AdaptiveStrategy
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced components not available: {e}")
    ENHANCED_COMPONENTS_AVAILABLE = False

# ==========================
# Logging
# ==========================
LOG_PATH = os.getenv("TB_TRADING_LOG", "trading_agent.log")
logger = logging.getLogger("hybrid_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

# ==========================
# Env/Safety Flags
# ==========================
OFFLINE = os.getenv("TB_TRADER_OFFLINE", "1") in ("1", "true", "on", "yes")
NO_TRADE = os.getenv("TB_NO_TRADE", "1") in ("1", "true", "on", "yes")
NOTIFY = os.getenv("TB_TRADER_NOTIFY", "0") == "1"
ENABLE_DISCORD = os.getenv("TB_ENABLE_DISCORD", "0") == "1"
NO_TELEGRAM = os.getenv("TB_NO_TELEGRAM", "1").lower() in ("1", "true", "on", "yes")
HEARTBEAT = os.getenv("TB_TRADER_NOTIFY_HEARTBEAT", "0") == "1"
HEARTBEAT_EVERY_N = int(os.getenv("TB_HEARTBEAT_EVERY_N", "12"))  # every N runs

# Live auto-apply of promoted parameters (opt-in)
AUTO_APPLY_ENABLED = os.getenv("TB_AUTO_APPLY_ENABLED", "0").lower() in ("1", "true", "on", "yes")
AUTO_APPLY_KILL = os.getenv("TB_AUTO_APPLY_KILL", "0").lower() in ("1", "true", "on", "yes")

# ==========================
# Per-fill logging helpers
# ==========================
GATE_MODE = os.getenv("TB_GATE_MODE", "normal")

def log_order_event(event: str, symbol: str, side: str, qty, price=None, extra=None):
    """
    Structured log for stderr/file so it shows in trader_loop.err and trading_agent.log.
    event: 'order_submitted' | 'order_filled' | 'order_partially_filled'
    """
    try:
        q = float(qty) if qty is not None else None
    except Exception:
        q = qty
    payload: Dict[str, Any] = {
        "event": event,
        "mode": GATE_MODE,
        "symbol": symbol,
        "side": side,
        "qty": q,
    }
    if price is not None:
        try:
            payload["price"] = float(price)
        except Exception:
            payload["price"] = price
    if extra:
        try:
            payload.update(extra)
        except Exception:
            pass
    logger.info("[order] %s", payload)
    
def try_log_fill_once(api: Optional[REST], order_id: Optional[str], symbol: str, side: str) -> None:
    """Best-effort single status check to emit a fill log without blocking the loop.
    Controlled by TB_LOG_FILLS=1. Safe if REST or order_id is missing.
    """
    if os.getenv("TB_LOG_FILLS", "1") != "1":
        return
    if api is None or not order_id:
        return
    try:
        def _get():
            return api.get_order(order_id)
        def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            # keep quiet; fills polling is best-effort
            return
        odr = retry_call(
            _get,
            attempts=1,
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry,
        )
        status = str(getattr(odr, "status", "")).lower()
        filled_qty = getattr(odr, "filled_qty", None)
        filled_avg_price = getattr(odr, "filled_avg_price", None)
        if status == "filled":
            log_order_event(
                "order_filled",
                symbol=symbol,
                side=side,
                qty=filled_qty,
                price=filled_avg_price,
                extra={"order_id": order_id},
            )
        elif status in ("partially_filled", "partial"):  # tolerate variants
            log_order_event(
                "order_partially_filled",
                symbol=symbol,
                side=side,
                qty=filled_qty,
                price=filled_avg_price,
                extra={"order_id": order_id},
            )
    except Exception:
        # best-effort only
        return
PROMOTED_PARAMS_PATH = Path("config/promoted_params.json")
AUTO_APPLY_AUDIT_DIR = Path("eval_runs/live_auto_apply")

# Optional ML gate
USE_ML_GATE = os.getenv("TB_USE_ML_GATE", "0") == "1"
ML_MODEL_PATH = os.getenv("TB_ML_GATE_MODEL_PATH", "eval_runs/ml/latest/model.pt")
ML_FEATURES_PATH = os.getenv("TB_ML_FEATURES_PATH", "eval_runs/ml/latest/features.csv")
ML_MIN_PROB = float(os.getenv("TB_ML_GATE_MIN_PROB", "0.5"))
ML_SOFT_GATE = os.getenv("TB_ML_GATE_SOFT", "1") == "1"

# Signal debounce: require EMA12>EMA26 for last N bars (0 disables)
SIGNAL_DEBOUNCE_N = int(os.getenv("TB_SIGNAL_DEBOUNCE_N", "1"))

# Optional volatility and higher-timeframe regime filters
USE_ATR_FILTER = os.getenv("TB_USE_ATR_FILTER", "0") == "1"
ATR_LEN = int(os.getenv("TB_ATR_LEN", "14"))
ATR_MIN_PCT = float(os.getenv("TB_ATR_MIN_PCT", "0.0"))
ATR_MAX_PCT = float(os.getenv("TB_ATR_MAX_PCT", "1.0"))

USE_HTF_REGIME = os.getenv("TB_USE_HTF_REGIME", "0") == "1"
HTF_EMA_LEN = int(os.getenv("TB_HTF_EMA_LEN", "200"))  # 1h EMA200 as 4h regime proxy

# Optional 1h entry path (in addition to 15m). When enabled, a 1h EMA cross-up
# can trigger a smaller-sized entry even if 15m cross is quiet.
USE_1H_ENTRY = os.getenv("TB_USE_1H_ENTRY", "1") == "1"
EMA_1H_FAST = int(os.getenv("TB_1H_EMA_FAST", "12"))
EMA_1H_SLOW = int(os.getenv("TB_1H_EMA_SLOW", "26"))
DEBOUNCE_1H_N = int(os.getenv("TB_1H_DEBOUNCE_N", "1"))
SIZE_1H_MULT = float(os.getenv("TB_1H_SIZE_MULT", "0.5"))  # fraction of normal size

# Multi-asset support
SYMBOL = os.getenv("SYMBOL", settings.symbol or "BTC/USD")
MULTI_ASSET_MODE = os.getenv("TB_MULTI_ASSET", "0") == "1"
ASSET_LIST = os.getenv("TB_ASSET_LIST", "BTC/USD,ETH/USD,SOL/USD,LINK/USD,LTC/USD,BCH/USD,UNI/USD,AAVE/USD,AVAX/USD,DOT/USD,MATIC/USD").split(",") if MULTI_ASSET_MODE else [SYMBOL]

# Alpaca-supported crypto pairs only (verified compatibility)
SUPPORTED_ASSETS = {
    # Major Blue Chips (Top Market Cap)
    "BTC/USD": {"min_size": 0.0001, "enabled": True},    # Bitcoin
    "ETH/USD": {"min_size": 0.001, "enabled": True},     # Ethereum
    "SOL/USD": {"min_size": 0.01, "enabled": True},      # Solana
    "LINK/USD": {"min_size": 0.1, "enabled": True},      # Chainlink
    
    # Additional Blue Chips (High Liquidity & Market Cap)
    "LTC/USD": {"min_size": 0.01, "enabled": True},      # Litecoin
    "BCH/USD": {"min_size": 0.001, "enabled": True},     # Bitcoin Cash
    "UNI/USD": {"min_size": 0.1, "enabled": True},       # Uniswap
    "AAVE/USD": {"min_size": 0.01, "enabled": True},     # Aave
    "AVAX/USD": {"min_size": 0.01, "enabled": True},     # Avalanche
    "DOT/USD": {"min_size": 0.1, "enabled": True},       # Polkadot
    "MATIC/USD": {"min_size": 1.0, "enabled": True},     # Polygon
    
    # DeFi Blue Chips
    "MKR/USD": {"min_size": 0.001, "enabled": True},     # Maker
    "COMP/USD": {"min_size": 0.01, "enabled": True},     # Compound
    "YFI/USD": {"min_size": 0.0001, "enabled": True},    # Yearn Finance
    "CRV/USD": {"min_size": 1.0, "enabled": True},       # Curve
    "SNX/USD": {"min_size": 0.1, "enabled": True},       # Synthetix
    "SUSHI/USD": {"min_size": 0.1, "enabled": True},     # SushiSwap
    
    # Additional Quality Projects
    "XTZ/USD": {"min_size": 0.1, "enabled": True},       # Tezos
    "GRT/USD": {"min_size": 1.0, "enabled": True},       # The Graph
    
    # Notes: 
    # - ADA/USD is NOT supported by Alpaca
    # - Min sizes are estimated based on typical Alpaca requirements
    # - All major blue chips with good liquidity included
}

TF_FAST = "15Min"
TF_SLOW = "1Hour"

# Risk and brackets
MAX_PORTFOLIO_RISK = float(os.getenv("TB_MAX_RISK_FRAC", "0.01"))   # 1%
TP_PCT = float(os.getenv("TB_TP_PCT", "0.05"))                      # +5%
SL_PCT = float(os.getenv("TB_SL_PCT", "0.02"))                      # -2%
DAILY_LOSS_CAP_PCT = float(os.getenv("TB_DAILY_LOSS_CAP_PCT", "0.03"))  # 3% of reference equity

# Enhanced Risk Management
USE_ENHANCED_RISK = os.getenv("TB_USE_ENHANCED_RISK", "1") == "1" and ENHANCED_COMPONENTS_AVAILABLE
USE_KELLY_SIZING = os.getenv("TB_USE_KELLY_SIZING", "1") == "1" and ENHANCED_COMPONENTS_AVAILABLE
USE_REGIME_DETECTION = os.getenv("TB_USE_REGIME_DETECTION", "1") == "1" and ENHANCED_COMPONENTS_AVAILABLE
USE_ENSEMBLE_ML = os.getenv("TB_USE_ENSEMBLE_ML", "1") == "1" and ENHANCED_COMPONENTS_AVAILABLE
USE_ADAPTIVE_STRATEGY = os.getenv("TB_USE_ADAPTIVE_STRATEGY", "1") == "1" and ENHANCED_COMPONENTS_AVAILABLE

# Portfolio Management
MAX_POSITIONS = int(os.getenv("TB_MAX_POSITIONS", "4"))
MAX_CORRELATION = float(os.getenv("TB_MAX_CORRELATION", "0.7"))
PORTFOLIO_VAR_LIMIT = float(os.getenv("TB_PORTFOLIO_VAR_LIMIT", "0.02"))

# Optional ATR-based stop sizing (replaces fixed SL_PCT when enabled)
USE_ATR_STOP = os.getenv("TB_USE_ATR_STOP", "0") == "1"
ATR_STOP_MULT = float(os.getenv("TB_ATR_STOP_MULT", "1.5"))

# Sentiment
SENTIMENT_THRESHOLD = float(os.getenv("TB_SENTIMENT_CUTOFF", "0.5"))
PPLX_TIMEOUT = float(os.getenv("TB_PPLX_TIMEOUT", "12"))

# Validation Mode Settings
VALIDATION_MODE = os.getenv("TB_VALIDATION_MODE", "0") == "1"
LOG_ALL_SIGNALS = os.getenv("TB_LOG_ALL_SIGNALS", "0") == "1"
MIN_CONFIDENCE = float(os.getenv("TB_MIN_CONFIDENCE", "0.65"))
DIVERGENCE_THRESHOLD = float(os.getenv("TB_DIVERGENCE_THRESHOLD", "0.5"))

# State/cooldown
COOLDOWN_SEC = int(os.getenv("TB_TRADER_COOLDOWN_SEC", "3600"))
STATE_DIR = Path("state")
RUNS_DIR = Path("runs")

# ==========================
# Enhanced Components Initialization
# ==========================

def initialize_enhanced_components():
    """Initialize enhanced trading components"""
    components = {}
    
    if not ENHANCED_COMPONENTS_AVAILABLE:
        logger.info("Enhanced components not available, using basic trading")
        return components
    
    try:
        # Initialize Advanced Risk Manager
        if USE_ENHANCED_RISK:
            components['risk_manager'] = AdvancedRiskManager()
            # Update limits
            components['risk_manager'].risk_limits.portfolio_var_limit = PORTFOLIO_VAR_LIMIT
            components['risk_manager'].risk_limits.max_correlation = MAX_CORRELATION
            logger.info("✅ Advanced Risk Manager initialized")
        
        # Initialize Kelly Position Sizer
        if USE_KELLY_SIZING:
            components['kelly_sizer'] = KellyPositionSizer()
            logger.info("✅ Kelly Position Sizer initialized")
        
        # Initialize Market Regime Detector
        if USE_REGIME_DETECTION:
            components['regime_detector'] = MarketRegimeDetector()
            logger.info("✅ Market Regime Detector initialized")
        
        # Initialize Ensemble ML Models
        if USE_ENSEMBLE_ML:
            try:
                components['ml_ensemble'] = TradingEnsemble()
                # Try to load a trained model if available
                model_path = "eval_runs/ml/latest/ensemble_model.pt"
                if os.path.exists(model_path):
                    components['ml_ensemble'].load_model(model_path)
                    logger.info("✅ Ensemble ML Models loaded from %s", model_path)
                else:
                    logger.info("✅ Ensemble ML Models initialized (no pretrained model)")
            except Exception as e:
                logger.warning("Failed to initialize ML ensemble: %s", e)
        
        # Initialize Adaptive Strategy
        if USE_ADAPTIVE_STRATEGY:
            components['adaptive_strategy'] = AdaptiveStrategy()
            logger.info("✅ Adaptive Strategy initialized")
            
        # Initialize Validation Tools
        if VALIDATION_TOOLS_AVAILABLE and VALIDATION_MODE:
            components['validation_analyzer'] = ValidationAnalyzer()
            components['threshold_optimizer'] = PaperTradingOptimizer()
            logger.info("✅ Validation tools initialized (validation mode)")
        elif VALIDATION_TOOLS_AVAILABLE:
            logger.info("✅ Validation tools available but validation mode disabled")
            
    except Exception as e:
        logger.error("Failed to initialize enhanced components: %s", e)
    
    return components

# ==========================
# Multi-Asset Trading Functions
# ==========================

def get_enabled_assets():
    """Get list of enabled assets for trading"""
    if MULTI_ASSET_MODE:
        # Filter to only supported assets
        enabled = []
        for asset in ASSET_LIST:
            asset = asset.strip()
            if asset in SUPPORTED_ASSETS and SUPPORTED_ASSETS[asset]["enabled"]:
                enabled.append(asset)
        return enabled
    else:
        # Single asset mode
        if SYMBOL in SUPPORTED_ASSETS and SUPPORTED_ASSETS[SYMBOL]["enabled"]:
            return [SYMBOL]
        else:
            logger.warning("Asset %s not in supported list, using anyway", SYMBOL)
            return [SYMBOL]

def calculate_enhanced_position_size(components, symbol, bars_15, bars_1h, sentiment, current_equity):
    """Calculate position size using enhanced methods"""
    base_size = current_equity * MAX_PORTFOLIO_RISK
    
    # Use Kelly Criterion if available
    if 'kelly_sizer' in components and 'regime_detector' in components:
        try:
            # Get current regime
            regime = components['regime_detector'].classify_regime({symbol: bars_1h})
            regime_str = components['regime_detector'].get_regime_string(regime)
            
            # Estimate win probability and win/loss ratio based on sentiment and regime
            win_prob = estimate_win_probability(sentiment, regime_str)
            win_loss_ratio = estimate_win_loss_ratio(regime_str)
            
            # Calculate Kelly size
            kelly_size = components['kelly_sizer'].calculate_kelly_size(
                win_prob, win_loss_ratio, current_equity, regime_str
            )
            
            logger.info("[%s] Kelly sizing: win_prob=%.3f, win_loss=%.2f, kelly_size=%.2f", 
                       symbol, win_prob, win_loss_ratio, kelly_size)
            
            return min(kelly_size, base_size * 2)  # Cap at 2x base size for safety
            
        except Exception as e:
            logger.warning("Kelly sizing failed for %s: %s", symbol, e)
    
    return base_size

def estimate_win_probability(sentiment, regime_str):
    """Estimate win probability based on sentiment and regime"""
    base_prob = 0.55
    
    # Handle None sentiment gracefully
    if sentiment is None:
        sentiment = 0.5  # Neutral default
    
    # Adjust for sentiment (stronger sentiment = higher confidence)
    sentiment_adj = (sentiment - 0.5) * 0.1
    base_prob += sentiment_adj
    
    # Adjust for regime
    if 'trending' in regime_str.lower():
        base_prob += 0.05
    elif 'volatile' in regime_str.lower():
        base_prob -= 0.05
    
    return np.clip(base_prob, 0.4, 0.75)

def estimate_win_loss_ratio(regime_str):
    """Estimate win/loss ratio based on regime"""
    base_ratio = 2.0
    
    if 'volatile' in regime_str.lower():
        base_ratio *= 0.8  # Lower ratio in volatile markets
    elif 'trending' in regime_str.lower():
        base_ratio *= 1.2  # Higher ratio in trending markets
    
    return base_ratio

def should_trade_asset(components, symbol, bars_15, bars_1h, sentiment, positions):
    """Enhanced decision logic for whether to trade an asset"""
    
    # Basic checks first
    if len(bars_15) < 50 or len(bars_1h) < 60:
        return False, "Insufficient data"
    
    # Check if we already have a position
    if symbol in positions:
        return False, "Already have position"
    
    # Portfolio limit check
    if len(positions) >= MAX_POSITIONS:
        return False, "Max positions reached"
    
    # Enhanced checks if components available
    if 'risk_manager' in components:
        try:
            # Get current prices for correlation check
            current_prices = {s: bars_1h['close'].iloc[-1] for s in positions.keys()}
            current_prices[symbol] = bars_1h['close'].iloc[-1]
            
            # Check portfolio limits
            limits_check = components['risk_manager'].check_portfolio_limits(
                positions, current_prices, (symbol, 1.0)  # Dummy size for check
            )
            
            if not all(limits_check.values()):
                failed_checks = [k for k, v in limits_check.items() if not v]
                return False, f"Portfolio limits failed: {failed_checks}"
                
        except Exception as e:
            logger.warning("Portfolio limits check failed: %s", e)
    
    # ML ensemble check if available
    if 'ml_ensemble' in components and USE_ML_GATE:
        try:
            # Build feature vector (simplified)
            features = build_ml_features(bars_15, bars_1h, sentiment)
            
            # Get ML prediction
            with torch.no_grad():
                prediction = components['ml_ensemble'](features.unsqueeze(0))
                confidence = torch.sigmoid(prediction).item()
            
            ml_threshold = float(os.getenv("TB_ML_GATE_MIN_PROB", "0.6"))
            if confidence < ml_threshold:
                return False, f"ML confidence too low: {confidence:.3f} < {ml_threshold}"
                
            logger.info("[%s] ML confidence: %.3f ✅", symbol, confidence)
            
        except Exception as e:
            logger.warning("ML prediction failed for %s: %s", symbol, e)
    
    return True, "All checks passed"

def calculate_atr(bars, period=14):
    """Calculate Average True Range"""
    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values
    
    # True Range calculation
    tr = np.maximum(high - low, 
                   np.maximum(abs(high - np.roll(close, 1)), 
                             abs(low - np.roll(close, 1))))
    
    # ATR is the moving average of True Range
    atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
    return atr if not np.isnan(atr) else 0.01

def build_ml_features(bars_15, bars_1h, sentiment):
    """Build feature vector for ML model"""
    # Handle None sentiment gracefully
    if sentiment is None:
        sentiment = 0.5
    
    # This is a simplified version - you can expand this
    features = []
    
    # Price-based features
    close_15 = bars_15['close'].values
    close_1h = bars_1h['close'].values
    
    # Returns
    returns_15 = np.diff(np.log(close_15[-20:]))  # Last 20 15min returns
    returns_1h = np.diff(np.log(close_1h[-10:]))  # Last 10 hourly returns
    
    # Simple statistics
    features.extend([
        np.mean(returns_15), np.std(returns_15),
        np.mean(returns_1h), np.std(returns_1h),
        sentiment,
        close_15[-1] / close_15[-20] - 1,  # 20-period return
        close_1h[-1] / close_1h[-10] - 1,  # 10-period return
    ])
    
    # Pad to expected size (37 features as in the enhanced agent)
    while len(features) < 37:
        features.append(0.0)
    
    return torch.tensor(features[:37], dtype=torch.float32)

# ==========================
# Helpers
# ==========================

def _rest() -> REST:
    return REST(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_base_url,
    )


def _normalize_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    if len(sym) >= 6:
        return f"{sym[:-3]}/{sym[-3:]}"
    return sym


def _decimals_for(sym: str) -> int:
    # Reasonable default for BTC/USD
    return 2 if "USD" in sym.replace("/", "") else 6


def _state_key_for(sym: str) -> str:
    s = _normalize_symbol(sym).replace("/", "-")
    return f"hybrid_trader_state_{s}.json"


def load_state(symbol: str) -> Dict[str, Any]:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        p = STATE_DIR / _state_key_for(symbol)
        if not p.exists():
            return {}
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(symbol: str, st: Dict[str, Any]) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        p = STATE_DIR / _state_key_for(symbol)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(st, f, indent=2, sort_keys=True)
        tmp.replace(p)
    except Exception as e:
        logger.warning(f"[state] save failed: {e}")


def _nowstamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
        tmp.replace(path)
    except Exception as e:
        logger.warning(f"[audit] write failed: {e}")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def maybe_auto_apply_params(now_utc: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """If enabled and not killed, load promoted params and apply to runtime globals.

    Maps promoted config fields to live runtime variables:
      - risk_frac -> MAX_PORTFOLIO_RISK
      - stop_mode == fixed_pct -> USE_ATR_STOP=False, TP_PCT/SL_PCT from tp_pct/sl_pct
      - stop_mode in {atr_fixed, atr_trailing} -> USE_ATR_STOP=True, ATR_STOP_MULT from atr_mult, ATR_LEN from atr_period, TP_PCT from tp_pct

    Writes an audit JSON under eval_runs/live_auto_apply/.
    """
    # Declare globals up-front since we both read and assign them
    global MAX_PORTFOLIO_RISK, TP_PCT, SL_PCT, USE_ATR_STOP, ATR_STOP_MULT, ATR_LEN
    if not AUTO_APPLY_ENABLED or AUTO_APPLY_KILL:
        return None
    try:
        if not PROMOTED_PARAMS_PATH.exists():
            return None
        with PROMOTED_PARAMS_PATH.open("r") as f:
            cfg = json.load(f)
        applied: Dict[str, Any] = {"status": "skipped", "reason": "no_changes"}
        # Prepare current snapshot
        curr = {
            "MAX_PORTFOLIO_RISK": MAX_PORTFOLIO_RISK,
            "TP_PCT": TP_PCT,
            "SL_PCT": SL_PCT,
            "USE_ATR_STOP": USE_ATR_STOP,
            "ATR_STOP_MULT": ATR_STOP_MULT,
            "ATR_LEN": ATR_LEN,
        }
        # Compute next
        next_vals = dict(curr)
        risk_frac = _safe_float(cfg.get("risk_frac"), curr["MAX_PORTFOLIO_RISK"])
        stop_mode = str(cfg.get("stop_mode") or "").strip()
        tp_pct = cfg.get("tp_pct")
        sl_pct = cfg.get("sl_pct")
        atr_mult = cfg.get("atr_mult")
        atr_period = cfg.get("atr_period")
        if risk_frac is not None:
            next_vals["MAX_PORTFOLIO_RISK"] = float(max(0.0, min(0.05, risk_frac)))
        if stop_mode == "fixed_pct":
            next_vals["USE_ATR_STOP"] = False
            if tp_pct is not None:
                next_vals["TP_PCT"] = max(0.0, float(tp_pct))
            if sl_pct is not None:
                next_vals["SL_PCT"] = max(0.0, float(sl_pct))
        elif stop_mode in ("atr_fixed", "atr_trailing"):
            next_vals["USE_ATR_STOP"] = True
            if atr_mult is not None:
                next_vals["ATR_STOP_MULT"] = max(0.1, float(atr_mult))
            if atr_period is not None:
                try:
                    next_vals["ATR_LEN"] = max(1, int(atr_period))
                except Exception:
                    pass
            if tp_pct is not None:
                next_vals["TP_PCT"] = max(0.0, float(tp_pct))
        # Detect changes
        changed = {k: (curr[k], next_vals[k]) for k in curr.keys() if curr[k] != next_vals[k]}
        if not changed:
            return None
        # Apply to globals
        MAX_PORTFOLIO_RISK = float(next_vals["MAX_PORTFOLIO_RISK"])
        TP_PCT = float(next_vals["TP_PCT"])
        SL_PCT = float(next_vals["SL_PCT"]) if not next_vals["USE_ATR_STOP"] else SL_PCT
        USE_ATR_STOP = bool(next_vals["USE_ATR_STOP"])
        ATR_STOP_MULT = float(next_vals["ATR_STOP_MULT"]) if USE_ATR_STOP else ATR_STOP_MULT
        ATR_LEN = int(next_vals["ATR_LEN"]) if USE_ATR_STOP else ATR_LEN
        # Audit
        ts = now_utc or datetime.now(timezone.utc).isoformat()
        audit = {
            "ts_utc": ts,
            "status": "applied",
            "from": curr,
            "to": next_vals,
            "promoted": cfg,
        }
        AUTO_APPLY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(AUTO_APPLY_AUDIT_DIR / f"apply_{_nowstamp()}.json", audit)
        logger.info("[auto_apply] Applied promoted params: %s", {k: v[1] for k, v in changed.items()})
        return audit
    except Exception as e:
        try:
            AUTO_APPLY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            write_json(AUTO_APPLY_AUDIT_DIR / f"apply_error_{_nowstamp()}.json", {"error": str(e)[:500]})
        except Exception:
            pass
        logger.warning(f"[auto_apply] failed: {e}")
        return None


def fetch_bars(symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    """Fetch recent crypto bars for given timeframe using Alpaca v2 crypto bars."""
    if OFFLINE:
        return synthetic_bars(timeframe, lookback)
    api = _rest()
    sym = _normalize_symbol(symbol)
    end = datetime.now(timezone.utc)
    # add buffer bars
    if timeframe == "1Min":
        delta = timedelta(minutes=lookback + 5)
    elif timeframe == "15Min":
        delta = timedelta(minutes=(lookback + 4) * 15)
    elif timeframe == "1Hour":
        delta = timedelta(hours=lookback + 2)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    start = end - delta
    def _get_bars():
        return api.get_crypto_bars(sym, timeframe, start.isoformat(), end.isoformat())
    def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
        logger.warning(f"[retry] get_bars attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
    bars_resp = retry_call(
        _get_bars,
        attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
        retry_exceptions=(Exception,),
        retry_status_codes=RETRY_STATUS_CODES,
        on_retry=_on_retry,
    )
    bars = bars_resp.df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(sym, level=0)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars = bars.tz_convert("UTC")
    # keep only required columns
    bars = bars[["open", "high", "low", "close", "volume"]].copy()
    bars.sort_index(inplace=True)
    return bars


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def detect_cross_up(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    f0, f1 = fast.iloc[-2], fast.iloc[-1]
    s0, s1 = slow.iloc[-2], slow.iloc[-1]
    return f0 <= s0 and f1 > s1


def detect_cross_down(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    f0, f1 = fast.iloc[-2], fast.iloc[-1]
    s0, s1 = slow.iloc[-2], slow.iloc[-1]
    return f0 >= s0 and f1 < s1


def one_hour_trend_up(bars_1h: pd.DataFrame) -> bool:
    ema50 = ema(bars_1h["close"], 50)
    return bool(bars_1h["close"].iloc[-1] > ema50.iloc[-1])

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


# ==========================
# Perplexity Sentiment
# ==========================

def _pplx_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


@CircuitBreaker(failure_threshold=3, recovery_timeout=60)
def sentiment_via_perplexity(headlines: list[str]) -> Tuple[float, Optional[str]]:
    """
    Ask Perplexity to score sentiment in [0,1] for Bitcoin given the provided headlines.
    Rotates through settings.pplx_api_keys. Returns (score, error_or_none).
    """
    if OFFLINE:
        # Deterministic mock sentiment in offline mode; no network calls
        return 0.62, None
    keys = settings.pplx_api_keys or []
    if not keys:
        return 0.0, "No Perplexity API keys"
    # Build a compact prompt; require strict JSON
    system_msg = (
        "You are an analysis API. Respond ONLY with a JSON object: {\"sentiment\": <float 0..1>}"
    )
    joined = "\n".join([h.strip() for h in headlines if h.strip()])[:2000]
    user_msg = (
        "Given these recent Bitcoin headlines, output a single scalar sentiment score in [0,1] "
        "where 0 is strongly bearish and 1 is strongly bullish.\n"
        f"Headlines:\n{joined}"
    )
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "web_search_options": {"enable_citation": False},
    }
    last_err: Optional[str] = None
    with httpx.Client(timeout=PPLX_TIMEOUT) as client:
        for key in keys:
            try:
                def _post() -> httpx.Response:
                    return client.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers=_pplx_headers(key),
                        json=payload,
                    )
                def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
                    logger.warning(
                        f"[retry] pplx attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s"
                    )
                r = retry_call(
                    _post,
                    attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
                    retry_exceptions=(httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError),
                    retry_status_codes=RETRY_STATUS_CODES,
                    on_retry=_on_retry,
                )
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code} {r.text[:160]}"; continue
                data = r.json()
                content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                text = str(content or "").strip()
                # Extract JSON object
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines and lines[0].startswith("```"): lines = lines[1:]
                    if lines and lines[-1].startswith("```"): lines = lines[:-1]
                    text = "\n".join(lines)
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    obj = {}
                    try:
                        import json
                        obj = json.loads(text[start:end+1])
                    except Exception:
                        pass
                    val = obj.get("sentiment") if isinstance(obj, dict) else None
                    try:
                        score = float(val)
                        if 0.0 <= score <= 1.0:
                            return score, None
                    except Exception:
                        pass
                last_err = "Invalid JSON response"
            except Exception as e:
                last_err = str(e)
    return 0.0, last_err


# ==========================
# Execution
# ==========================

class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

def _load_ml_gate(model_path: str, features_path: str):
    try:
        import pandas as _pd
        feats = _pd.read_csv(features_path)["feature"].tolist()
        model = _MLP(len(feats))
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, feats
    except Exception as e:
        logger.warning(f"[ml_gate] load failed: {e}")
        return None, None

def _build_live_feature_vector(bars_15: pd.DataFrame, feature_names: list[str]) -> Optional[torch.Tensor]:
    try:
        df = bars_15.copy()

        # Basic returns
        df["ret_1"] = df["close"].pct_change()

        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI divergence (difference between current RSI and its SMA)
        df["rsi_sma"] = df["rsi"].rolling(window=14).mean()
        df["rsi_divergence"] = df["rsi"] - df["rsi_sma"]

        # MACD (Moving Average Convergence Divergence)
        df["ema12"] = ema(df["close"], 12)
        df["ema26"] = ema(df["close"], 26)
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = ema(df["macd"], 9)
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # MACD momentum (rate of change of MACD histogram)
        df["macd_momentum"] = df["macd_histogram"].pct_change()

        # Bollinger Bands
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["std20"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["sma20"] + (df["std20"] * 2)
        df["bb_lower"] = df["sma20"] - (df["std20"] * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Bollinger Band width (volatility measure)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma20"]

        # Momentum indicators
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

        # Rate of Change (ROC) indicators
        df["roc_5"] = df["close"].pct_change(periods=5)
        df["roc_10"] = df["close"].pct_change(periods=10)
        df["roc_20"] = df["close"].pct_change(periods=20)

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()

        # Volatility measures
        df["close_volatility"] = df["close"].pct_change().rolling(20).std()
        df["high_low_range"] = (df.get("high", df["close"]) - df.get("low", df["close"])) / df["close"]

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        df["volume_trend"] = df["volume"].pct_change().rolling(5).mean()

        # Price acceleration (second derivative)
        df["price_acceleration"] = df["close"].pct_change().pct_change()

        # Support and resistance levels (simplified)
        df["support_level"] = df["low"].rolling(window=20).min()
        df["resistance_level"] = df["high"].rolling(window=20).max()
        df["support_distance"] = (df["close"] - df["support_level"]) / df["close"]
        df["resistance_distance"] = (df["resistance_level"] - df["close"]) / df["close"]
        
        # Combined support-resistance feature for ML model
        df["support_resistance"] = df["support_distance"] - df["resistance_distance"]

        # Original features
        df["ema12_slope"] = df["ema12"].pct_change()
        df["ema26_slope"] = df["ema26"].pct_change()
        df["vol"] = df["close"].pct_change().rolling(20).std().fillna(0.0)
        df["vol_chg"] = df["volume"].pct_change().fillna(0.0)
        df["cross_up"] = ((df["ema12"].shift(1) <= df["ema26"].shift(1)) & (df["ema12"] > df["ema26"]))
        df["cross_up"] = df["cross_up"].astype(int)

        # Align to the last fully-formed feature row (like training: features drop last row)
        df = df.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        # Use the penultimate row to mimic training alignment
        last_idx = -2 if len(df) >= 2 else -1
        try:
            vec = [float(df.iloc[last_idx][name]) for name in feature_names]
        except KeyError as e:
            logger.warning(f"[ml_gate] Missing feature: {e}, available: {list(df.columns)}")
            return None
        x = torch.tensor([vec], dtype=torch.float32)
        return x
    except Exception as e:
        logger.warning(f"[ml_gate] feature build failed: {e}")
        return None

def get_account_equity(api: REST) -> float:
    try:
        def _get_acct():
            return api.get_account()
        def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] get_account attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        acct = retry_call(
            _get_acct,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry,
        )
        eq = float(getattr(acct, "equity", getattr(acct, "cash", 0.0)) or 0.0)
        return eq
    except Exception:
        return 0.0


def reconcile_position_state(api: Optional[REST], symbol: str, st: Dict[str, Any]) -> Dict[str, Any]:
    """Update st['in_position'] based on broker positions when online; otherwise keep as-is."""
    if api is None:
        return st
    try:
        sym = _normalize_symbol(symbol)
        qty = 0.0
        for p in api.list_positions():
            if getattr(p, "symbol", "") == sym:
                try:
                    qty = abs(float(p.qty))
                except Exception:
                    qty = 0.0
                break
        st = dict(st)
        st["in_position"] = bool(qty > 0.0)
        return st
    except Exception as e:
        logger.warning(f"[state] reconcile failed: {e}")
        return st


def calc_position_size(equity: float, entry: float, stop: float) -> float:
    risk_amt = equity * MAX_PORTFOLIO_RISK
    per_unit_risk = max(entry - stop, 1e-9)
    qty = risk_amt / per_unit_risk
    # Enforce optional hard notional cap per trade
    try:
        cap_notional = float(os.getenv("TB_MAX_NOTIONAL_PER_TRADE", "0"))
    except Exception:
        cap_notional = 0.0
    if cap_notional > 0:
        max_qty = cap_notional / max(entry, 1e-9)
        qty = min(qty, max_qty)
    return max(0.0, float(qty))


def place_bracket(api: REST, symbol: str, qty: float, entry: float, tp: float, sl: float) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        # Enforce optional hard notional cap per trade at submission time as a safety net
        try:
            cap_notional = float(os.getenv("TB_MAX_NOTIONAL_PER_TRADE", "0"))
        except Exception:
            cap_notional = 0.0
        if cap_notional > 0 and entry > 0:
            try:
                qty = min(float(qty), cap_notional / float(entry))
            except Exception:
                pass
        def _submit():
            return api.submit_order(
                symbol=_normalize_symbol(symbol),
                side="buy",
                type="market",
                time_in_force="gtc",
                qty=qty,
                take_profit={"limit_price": round(tp, _decimals_for(symbol))},
                stop_loss={"stop_price": round(sl, _decimals_for(symbol))},
            )
        def _on_retry(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] submit_order attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        order = retry_call(
            _submit,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry,
        )
        oid = getattr(order, "id", None) or getattr(order, "client_order_id", None)
        # Per-fill logging: order submission
        try:
            log_order_event(
                "order_submitted",
                symbol=symbol,
                side="buy",
                qty=qty,
                extra={
                    "order_id": (str(oid) if oid else None),
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                },
            )
        except Exception:
            pass
        # Best-effort immediate fill check (non-blocking)
        try:
            try_log_fill_once(api, str(oid) if oid else None, symbol, "buy")
        except Exception:
            pass
        return True, (str(oid) if oid else None), None
    except Exception as e:
        return False, None, str(e)


def close_position_if_any(api: REST, symbol: str) -> Optional[str]:
    sym = _normalize_symbol(symbol)
    try:
        def _list_pos():
            return api.list_positions()
        def _on_retry_list(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] list_positions attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        positions = retry_call(
            _list_pos,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry_list,
        )
        pos = None
        for p in positions:
            if getattr(p, "symbol", "") == sym:
                pos = p; break
        if not pos:
            return None
        qty = abs(float(pos.qty))
        if qty <= 0:
            return None
        def _submit_close():
            return api.submit_order(symbol=sym, side="sell", type="market", time_in_force="gtc", qty=qty)
        def _on_retry_close(attempt: int, status: Optional[int], exc: Exception, sleep_s: float) -> None:
            logger.warning(f"[retry] submit_close attempt {attempt} status={status} err={str(exc)[:120]} next={sleep_s:.2f}s")
        order = retry_call(
            _submit_close,
            attempts=int(os.getenv("TB_RETRY_ATTEMPTS", "5")),
            retry_exceptions=(Exception,),
            retry_status_codes=RETRY_STATUS_CODES,
            on_retry=_on_retry_close,
        )
        oid = getattr(order, "id", None) or getattr(order, "client_order_id", None)
        # Per-fill logging: close order submission
        try:
            log_order_event(
                "order_submitted",
                symbol=sym,
                side="sell",
                qty=qty,
                extra={
                    "order_id": (str(oid) if oid else None),
                    "reason": "close_position",
                },
            )
        except Exception:
            pass
        # Best-effort immediate fill check (non-blocking)
        try:
            try_log_fill_once(api, str(oid) if oid else None, sym, "sell")
        except Exception:
            pass
        return str(oid) if oid else None
    except Exception as e:
        logger.warning(f"[trade] close error: {e}")
        return None


# ==========================
# Notifications
# ==========================

# Import Telegram sender from project root module
try:
    from telegram_bot import send_message as send_telegram  # type: ignore
except Exception as _e:
    # Keep module import-safe; notify() will guard and log if used
    send_telegram = None  # type: ignore

def send_discord_embed(webhook_url: str, embeds: list[dict]) -> bool:
    """Minimal Discord webhook sender. Returns True on HTTP 2xx, else False."""
    try:
        if not webhook_url:
            return False
        with httpx.Client(timeout=float(os.getenv("TB_DISCORD_TIMEOUT", "5"))) as client:
            resp = client.post(webhook_url, json={"embeds": embeds})
            if 200 <= resp.status_code < 300:
                return True
            logger.warning(f"[discord] status={resp.status_code} body={resp.text[:200]}")
            return False
    except Exception as e:
        logger.warning(f"[discord] error: {e}")
        return False

def notify(event: str, payload: Dict[str, Any]) -> None:
    if not NOTIFY:
        return
    # Tailor fields for readability depending on event type; keep TG/Discord parity
    et = event.lower()
    sym = payload.get("symbol", "")
    sent = payload.get("sentiment")
    status = payload.get("status")
    # Build message strings per event
    if et in ("submit", "would_submit"):
        qty = payload.get("qty")
        entry = payload.get("entry")
        tp = payload.get("tp")
        sl = payload.get("sl")
        price = payload.get("price")
        desc_lines = [
            f"tf_fast={TF_FAST} tf_slow={TF_SLOW}",
            f"qty={qty} entry={entry} tp={tp} sl={sl}",
            f"price={price} sentiment={sent}",
            f"status={status}",
        ]
        tg_msg = (
            f"Hybrid Trader • {sym}. {event}. "
            f"qty={qty} entry={entry} tp={tp} sl={sl} sentiment={sent}"
        )
        color = 0x2ecc71
    elif et in ("close", "would_close"):
        qty = payload.get("qty")
        entry = payload.get("entry")
        exit_px = payload.get("price")
        pnl_est = payload.get("pnl_est")
        desc_lines = [
            f"tf_fast={TF_FAST} tf_slow={TF_SLOW}",
            f"qty={qty} entry={entry} exit={exit_px} pnl_est={pnl_est}",
            f"sentiment={sent}",
            f"status={status}",
        ]
        tg_msg = (
            f"Hybrid Trader • {sym}. {event}. "
            f"qty={qty} entry={entry} exit={exit_px} pnl={pnl_est} sentiment={sent}"
        )
        color = 0x95a5a6
    else:  # heartbeat/others
        price = payload.get("price")
        desc_lines = [
            f"tf_fast={TF_FAST} tf_slow={TF_SLOW}",
            f"price={price} sentiment={sent}",
            f"status={status}",
        ]
        tg_msg = (
            f"Hybrid Trader • {sym}. {event}. "
            f"price={price} sentiment={sent} status={status}"
        )
        color = 0x95a5a6

    embed = {
        "title": f"Trader: {event} {sym}",
        "description": "\n".join(desc_lines),
        "color": color,
    }
    if ENABLE_DISCORD:
        try:
            webhook = os.getenv("DISCORD_TRADER_WEBHOOK_URL", "") or os.getenv("DISCORD_WEBHOOK_URL", "")
            if webhook:
                send_discord_embed(webhook, [embed])
        except Exception as e:
            logger.warning(f"[notify] discord error: {e}")
    if not NO_TELEGRAM:
        try:
            msg = tg_msg
            if send_telegram is not None:
                send_telegram(msg)
            else:
                logger.info("[notify] telegram module not available; skipped")
        except Exception as e:
            logger.warning(f"[notify] telegram error: {e}")


# ==========================
# Main loop (single-run)
# ==========================

def main() -> int:
    logger.info("Starting Enhanced Multi-Asset Hybrid Trader (safe=%s, no_trade=%s)", OFFLINE, NO_TRADE)
    
    # Initialize performance tracking
    performance_tracker.heartbeat()
    
    # Run health checks
    health_results = health_monitor.run_health_checks()
    health_summary = health_monitor.get_system_health_summary()
    logger.info("System health: %s (critical=%d warning=%d healthy=%d)", 
                health_summary['status'], 
                health_summary['summary']['critical'],
                health_summary['summary']['warning'], 
                health_summary['summary']['healthy'])
    
    # Initialize enhanced components
    components = initialize_enhanced_components()
    
    # Get enabled assets
    enabled_assets = get_enabled_assets()
    logger.info("Trading assets: %s", enabled_assets)
    
    # Assign a run_id early for consistent logging
    _run_id = _nowstamp()
    logger.info("[run] start run_id=%s assets=%s", _run_id, enabled_assets)
    
    # Log performance stats and configuration
    stats = performance_tracker.get_stats()
    config_summary = config_manager.get_config_summary()
    logger.info("Performance stats: uptime=%.1fh trades=%d errors=%d", 
                stats["uptime_hours"], stats["trade_count"], stats["error_count"])
    logger.info("Config summary: NO_TRADE=%s OFFLINE=%s MULTI_ASSET=%s", 
                config_summary.get('TB_NO_TRADE', 'NOT_SET'),
                config_summary.get('TB_OFFLINE', 'NOT_SET'), 
                config_summary.get('TB_MULTI_ASSET', 'NOT_SET'))
    
    # Log ML model info if available
    try:
        if USE_ML_GATE and ML_MODEL_PATH:
            _resolved_model_dir = os.path.dirname(os.path.realpath(ML_MODEL_PATH))
            logger.info("[ml_gate] using model_dir=%s", _resolved_model_dir)
    except Exception:
        pass
    
    # Auto-apply promoted parameters
    maybe_auto_apply_params()
    api = _rest() if not OFFLINE else None
    
    # Get current equity
    current_equity = get_account_equity(api) if not OFFLINE else 100000.0
    logger.info("Current equity: $%.2f", current_equity)
    
    # Load states for all assets and reconcile positions
    asset_states = {}
    current_positions = {}
    
    for symbol in enabled_assets:
        # Use SQLite state management instead of JSON
        state = trading_db.load_position_state(symbol)
        if not OFFLINE:
            state = reconcile_position_state(api, symbol, state)
        asset_states[symbol] = state
        
        # Track current positions
        if state.get("position_side") == "long":
            current_positions[symbol] = {
                'side': 'long',
                'qty': state.get("position_qty", 0),
                'entry_price': state.get("position_entry_price", 0),
                'timestamp': state.get("last_entry_time", "")
            }
    
    logger.info("Current positions: %s", list(current_positions.keys()))
    
    # Daily PnL book-keeping
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total_pnl_today = sum(state.get("pnl_today", 0) for state in asset_states.values())
    
    # Check daily loss limit
    daily_loss_limit = current_equity * DAILY_LOSS_CAP_PCT
    if total_pnl_today < -daily_loss_limit:
        logger.warning("Daily loss limit reached: $%.2f < -$%.2f", total_pnl_today, daily_loss_limit)
        return 0
    
    # Process each asset
    trading_decisions = []
    
    for symbol in enabled_assets:
        try:
            logger.info("[progress] Processing %s...", symbol)
            
            # Fetch data in parallel for better performance
            start_time = time.time()
            parallel_data = async_processor.fetch_all_symbol_data(symbol)
            fetch_time = time.time() - start_time
            
            bars_15 = parallel_data.get('bars_15m')
            bars_1h = parallel_data.get('bars_1h')
            sentiment = parallel_data.get('sentiment', 0.5)
            
            logger.debug(f"Parallel data fetch for {symbol}: {fetch_time:.2f}s")
            
            if (bars_15 is None or bars_1h is None or 
                len(bars_15) < 50 or len(bars_1h) < 60):
                logger.warning("Insufficient bars for %s: 15m=%d 1h=%d", symbol, 
                              len(bars_15) if bars_15 is not None else 0, 
                              len(bars_1h) if bars_1h is not None else 0)
                continue
            
            # Initialize signal variables to avoid UnboundLocalError
            cross_up = False
            cross_down = False
            cross_up_1h = False
            cross_down_1h = False
            trend_up = False
            price = 0.0
            
            # Calculate indicators
            ema12 = ema(bars_15["close"], 12)
            ema26 = ema(bars_15["close"], 26)
            ema50h = ema(bars_1h["close"], 50)
            ema1h_fast = ema(bars_1h["close"], EMA_1H_FAST)
            ema1h_slow = ema(bars_1h["close"], EMA_1H_SLOW)
            
            cross_up = detect_cross_up(ema12, ema26)
            cross_down = detect_cross_down(ema12, ema26)
            cross_up_1h = detect_cross_up(ema1h_fast, ema1h_slow)
            cross_down_1h = detect_cross_down(ema1h_fast, ema1h_slow)
            trend_up = bool(bars_1h["close"].iloc[-1] > ema50h.iloc[-1])
            
            price = float(bars_15["close"].iloc[-1])
            
            # Use pre-fetched sentiment from parallel processing
            if OFFLINE:
                headlines = [f"{symbol} consolidates after recent move; traders watch key levels"]
                sentiment = sentiment or 0.6  # Use fetched sentiment or default
            # Sentiment already fetched in parallel, use that value
            
            # Enhanced trading decision
            can_trade, reason = should_trade_asset(components, symbol, bars_15, bars_1h, sentiment, current_positions)
            
            if not can_trade:
                logger.info("[%s] Cannot trade: %s", symbol, reason)
                continue
            
            # Check basic entry conditions
            state = asset_states[symbol]
            last_entry_ts = state.get("last_entry_time", 0)
            # Handle None values from SQLite
            if last_entry_ts is None:
                last_entry_ts = 0
            cooldown_ok = (time.time() - last_entry_ts) > COOLDOWN_SEC
            
            # Entry logic
            entry_signal = False
            entry_reason = ""
            
            if cross_up and trend_up and sentiment > SENTIMENT_THRESHOLD and cooldown_ok:
                entry_signal = True
                entry_reason = "EMA cross up + trend up + positive sentiment"
            elif cross_up_1h and trend_up and sentiment > SENTIMENT_THRESHOLD and cooldown_ok:
                entry_signal = True
                entry_reason = "1H EMA cross up + trend up + positive sentiment"
            
            if entry_signal:
                # Ensure sentiment is not None before calculations
                if sentiment is None:
                    sentiment = 0.5  # Neutral default
                    logger.warning(f"Sentiment was None for {symbol}, using neutral default")
                
                # Calculate confidence and divergence for validation
                # Basic confidence calculation (can be enhanced)
                confidence = min(0.9, 0.5 + abs(sentiment - 0.5) + (0.1 if trend_up else 0))
                divergence = abs(sentiment - 0.5) * 2  # Simple divergence proxy
                
                # Log signal for validation tracking
                if VALIDATION_TOOLS_AVAILABLE and (VALIDATION_MODE or LOG_ALL_SIGNALS):
                    signal_data = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'symbol': symbol,
                        'signal_type': 'entry',
                        'entry_reason': entry_reason,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'divergence': divergence,
                        'price': price,
                        'cross_up': cross_up,
                        'cross_up_1h': cross_up_1h,
                        'trend_up': trend_up,
                        'cooldown_ok': cooldown_ok,
                        'will_trade': False,  # Will be updated below
                        'validation_mode': VALIDATION_MODE
                    }
                    
                    # Log signal to validation system
                    try:
                        if 'validation_analyzer' in components:
                            # Store for later analysis
                            components['validation_analyzer'].log_signal(signal_data)
                    except Exception as e:
                        logger.warning("Failed to log validation signal: %s", e)
                
                # Calculate position size using enhanced methods
                position_size_usd = calculate_enhanced_position_size(
                    components, symbol, bars_15, bars_1h, sentiment, current_equity
                )
                
                qty = position_size_usd / price
                min_size = SUPPORTED_ASSETS.get(symbol, {}).get("min_size", 0.001)
                
                if qty >= min_size:
                    # Calculate stop loss and take profit
                    if USE_ATR_STOP:
                        atr = calculate_atr(bars_15)
                        stop_loss = price - (atr * ATR_STOP_MULT)
                    else:
                        stop_loss = price * (1 - SL_PCT)
                    
                    take_profit = price * (1 + TP_PCT)
                    
                    decision = {
                        'symbol': symbol,
                        'action': 'BUY',
                        'qty': qty,
                        'price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': entry_reason,
                        'sentiment': sentiment,
                        'size_usd': position_size_usd
                    }
                    trading_decisions.append(decision)
                    
                    # Update validation signal to indicate trade will execute
                    if VALIDATION_TOOLS_AVAILABLE and (VALIDATION_MODE or LOG_ALL_SIGNALS):
                        signal_data['will_trade'] = True
                        signal_data['qty'] = qty
                        signal_data['size_usd'] = position_size_usd
                        signal_data['stop_loss'] = stop_loss
                        signal_data['take_profit'] = take_profit
                    
                    logger.info("[%s] ENTRY SIGNAL: %s (qty=%.4f, $%.2f)", 
                               symbol, entry_reason, qty, position_size_usd)
                else:
                    logger.info("[%s] Position size too small: %.6f < %.6f", symbol, qty, min_size)
            
            # Check exit conditions for existing positions
            if symbol in current_positions:
                position = current_positions[symbol]
                entry_price = position['entry_price']
                current_pnl_pct = (price - entry_price) / entry_price
                
                should_exit = False
                exit_reason = ""
                
                # Exit conditions
                if cross_down:
                    should_exit = True
                    exit_reason = "EMA cross down"
                elif cross_down_1h:
                    should_exit = True
                    exit_reason = "1H EMA cross down"
                elif current_pnl_pct <= -SL_PCT:
                    should_exit = True
                    exit_reason = "Stop loss hit"
                elif current_pnl_pct >= TP_PCT:
                    should_exit = True
                    exit_reason = "Take profit hit"
                elif sentiment < 0.3:
                    should_exit = True
                    exit_reason = "Negative sentiment"
                
                if should_exit:
                    decision = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'qty': position['qty'],
                        'price': price,
                        'reason': exit_reason,
                        'pnl_pct': current_pnl_pct
                    }
                    trading_decisions.append(decision)
                    
                    logger.info("[%s] EXIT SIGNAL: %s (PnL: %.2f%%)", 
                               symbol, exit_reason, current_pnl_pct * 100)
        
        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e)
            performance_tracker.record_error()
            continue
    
    # Execute trading decisions
    executed_trades = []
    for decision in trading_decisions:
        try:
            symbol = decision['symbol']
            
            if decision['action'] == 'BUY':
                if not NO_TRADE and api:
                    success, order_id, error = place_bracket(
                        api, symbol, decision['qty'], decision['price'],
                        decision['take_profit'], decision['stop_loss']
                    )
                    
                    if success:
                        # Track performance
                        performance_tracker.record_trade()
                        
                        # Log to database
                        trading_db.log_trade(
                            run_id=_run_id,
                            symbol=symbol,
                            action='BUY',
                            price=decision['price'],
                            confidence=decision.get('confidence', 0.5)
                        )
                        
                        # Update state
                        state = asset_states[symbol]
                        state.update({
                            "position_side": "long",
                            "position_qty": decision['qty'],
                            "position_entry_price": decision['price'],
                            "last_entry_time": time.time(),
                            "order_id": order_id
                        })
                        # Save state to both SQLite and JSON for now
                        trading_db.save_position_state(symbol, state)
                        save_state(symbol, state)
                        
                        executed_trades.append(decision)
                        logger.info("✅ Executed BUY for %s: %.4f @ $%.2f", 
                                   symbol, decision['qty'], decision['price'])
                    else:
                        logger.error("❌ Failed to execute BUY for %s: %s", symbol, error)
                else:
                    logger.info("🔒 Simulated BUY for %s: %.4f @ $%.2f", 
                               symbol, decision['qty'], decision['price'])
                    executed_trades.append(decision)
            
            elif decision['action'] == 'SELL':
                if not NO_TRADE and api:
                    order_id = close_position_if_any(api, symbol)
                    
                    if order_id:
                        # Update state
                        state = asset_states[symbol]
                        pnl_usd = (decision['price'] - state.get('position_entry_price', 0)) * decision['qty']
                        state.update({
                            "position_side": None,
                            "position_qty": 0,
                            "position_entry_price": 0,
                            "pnl_today": state.get("pnl_today", 0) + pnl_usd,
                            "last_exit_time": time.time()
                        })
                        save_state(symbol, state)
                        
                        executed_trades.append(decision)
                        logger.info("✅ Executed SELL for %s: PnL $%.2f (%.2f%%)", 
                                   symbol, pnl_usd, decision['pnl_pct'] * 100)
                    else:
                        logger.error("❌ Failed to execute SELL for %s", symbol)
                else:
                    logger.info("🔒 Simulated SELL for %s: PnL %.2f%%", 
                               symbol, decision['pnl_pct'] * 100)
                    executed_trades.append(decision)
        
        except Exception as e:
            logger.error("Failed to execute trade for %s: %s", decision['symbol'], e)
    
    # Send notifications
    if executed_trades and NOTIFY:
        try:
            message = f"Enhanced Multi-Asset Trader Update:\n"
            message += f"Processed {len(enabled_assets)} assets\n"
            message += f"Executed {len(executed_trades)} trades\n\n"
            
            for trade in executed_trades:
                if trade['action'] == 'BUY':
                    message += f"🟢 BUY {trade['symbol']}: {trade['qty']:.4f} @ ${trade['price']:.2f}\n"
                    message += f"   Reason: {trade['reason']}\n"
                    message += f"   Sentiment: {trade['sentiment']:.2f}\n\n"
                else:
                    message += f"🔴 SELL {trade['symbol']}: PnL {trade['pnl_pct']*100:.2f}%\n"
                    message += f"   Reason: {trade['reason']}\n\n"
            
            # Send notifications
            if not NO_TELEGRAM:
                send_telegram(message)
            
            if ENABLE_DISCORD:
                # Send to Discord (simplified)
                embed = {
                    "title": "Enhanced Multi-Asset Trader",
                    "description": message,
                    "color": 0x00ff00 if executed_trades else 0x808080
                }
                webhook = os.getenv("DISCORD_TRADER_WEBHOOK_URL", "") or os.getenv("DISCORD_WEBHOOK_URL", "")
                if webhook:
                    send_discord_embed(webhook, [embed])
            
        except Exception as e:
            logger.error("Failed to send notifications: %s", e)
    
    # Update adaptive strategy if available
    if 'adaptive_strategy' in components and executed_trades:
        try:
            for trade in executed_trades:
                if trade['action'] == 'SELL' and 'pnl_pct' in trade:
                    components['adaptive_strategy'].update_performance({
                        'symbol': trade['symbol'],
                        'pnl_pct': trade['pnl_pct'],
                        'timestamp': datetime.now()
                    })
        except Exception as e:
            logger.warning("Failed to update adaptive strategy: %s", e)
    
    # Run validation analysis if enabled
    if VALIDATION_TOOLS_AVAILABLE and VALIDATION_MODE:
        try:
            if 'validation_analyzer' in components:
                validation_report = components['validation_analyzer'].generate_validation_report()
                logger.info("📊 Validation Report: Signals/week=%.1f, Target=7.0", 
                           validation_report.get('signal_frequency', {}).get('current_avg_per_week', 0))
                
                # Log recommendations if any
                recommendations = validation_report.get('actionable_recommendations', {})
                if recommendations.get('immediate_actions'):
                    for action in recommendations['immediate_actions'][:2]:  # Show top 2
                        logger.info("💡 Validation Recommendation: %s", action)
        except Exception as e:
            logger.warning("Failed to run validation analysis: %s", e)
    
    logger.info("Enhanced multi-asset trading cycle complete: %d trades executed", len(executed_trades))
    
    # Enhanced mode always uses multi-asset processing
    decision = {"action": "multi_asset_complete", "trades": len(executed_trades)}
    
    # Define variables needed for heartbeat and completion
    now_ts = int(time.time())
    run_id = _run_id
    run_dir = None
    
    # Create a simple global state for heartbeat tracking (since we removed single-asset state)
    state = {
        "hb_runs": 0,
        "last_run_ts": now_ts,
        "last_heartbeat_ts": 0
    }
    
    # Skip old single-asset code - removed to prevent UnboundLocalError
    # The old single-asset code has been removed as we now use multi-asset processing exclusively
    
    # Heartbeat: per-run counter + optional liveness notification
    try:
        hb_runs = int(state.get("hb_runs", 0)) + 1
        state["hb_runs"] = hb_runs
        state["last_run_ts"] = now_ts
        if HEARTBEAT and NOTIFY and HEARTBEAT_EVERY_N > 0 and (hb_runs % HEARTBEAT_EVERY_N == 0):
            payload = {
                "symbol": "MULTI-ASSET",
                "price": 0,  # Multi-asset mode doesn't have single price
                "sentiment": 0,  # Multi-asset mode doesn't have single sentiment
                "qty": 0,
                "entry": None,
                "tp": None,
                "sl": None,
                "status": f"alive run={hb_runs} trades={len(executed_trades)}",
            }
            notify("heartbeat", payload)
            state["last_heartbeat_ts"] = now_ts
            logger.info("[heartbeat] sent run=%d every=%d", hb_runs, HEARTBEAT_EVERY_N)
    except Exception as e:
        logger.warning(f"[heartbeat] failed: {e}")

    # Persist global state for heartbeat tracking
    try:
        # Save global heartbeat state (not per-symbol since we're multi-asset)
        global_state_file = os.path.join(STATE_DIR, "global_state.json")
        write_json(global_state_file, state)
    except Exception:
        pass

    # Write decision + post-state
    if os.getenv("TB_AUDIT", "1") == "1" and run_dir is not None:
        try:
            write_json(run_dir / "decision.json", {
                "decision": decision,
                "state": state,
            })
            logger.info(f"[progress] Wrote audit decision -> {run_dir}/decision.json")
        except Exception as e:
            logger.warning(f"[audit] decision write failed: {e}")

    # Optional: auto-commit safe artifacts (non-code) using autocommit helper
    try:
        if os.getenv("TB_AUTOCOMMIT_ARTIFACTS", "1") == "1":
            push_enabled = os.getenv("TB_AUTOCOMMIT_PUSH", "1") == "1"
            # Call autocommit.auto_commit_and_push safely
            code = subprocess.call([
                "python3", "-c",
                (
                    "import autocommit as ac; "
                    "print(ac.auto_commit_and_push(['runs','eval_runs','universe_runs','bars','state','data','eval_data','validation_reports','trader_loop.log','trading_agent.log','validation_analysis.db','integration_test_results_*.json','test_integration_report.json','*_integration_report.json'], "
                    "extra_message='local artifacts + validation data + integration tests + bar data + trader state + eval data + validation reports', push_enabled="
                    + ("True" if push_enabled else "False") +
                    "))"
                )
            ])
            logger.info("[autocommit] attempted with push=%s status=%s", push_enabled, code)
    except Exception as e:
        logger.warning("[autocommit] failed: %s", e)
    # Final run-complete marker
    try:
        logger.info("[run] complete run_id=%s decision=%s", run_id, (decision or {}).get("action"))
    except Exception:
        pass
    return 0


# ==========================
# Offline helpers
# ==========================
def synthetic_bars(timeframe: str, lookback: int) -> pd.DataFrame:
    """
    Generate deterministic synthetic OHLCV series for preview without network calls.
    Creates a gentle uptrend with minor noise to allow cross/EMA computations.
    """
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if timeframe == "1Min":
        step = timedelta(minutes=1)
    elif timeframe == "15Min":
        step = timedelta(minutes=15)
    elif timeframe == "1Hour":
        step = timedelta(hours=1)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    n = max(lookback + 60, 120)  # ensure enough warmup for EMAs
    idx = [now - step * (n - i) for i in range(n)]
    base = 60000.0  # starting price
    drift = 0.0002  # per bar drift
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 5.0, size=n)
    close = base + np.cumsum(base * drift + noise)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.0, 10.0, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.0, 10.0, size=n)
    vol = rng.uniform(5, 50, size=n)
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=pd.DatetimeIndex(idx, tz="UTC"))
    df = df.iloc[-(lookback + 5):].copy()
    return df

if __name__ == "__main__":
    raise SystemExit(main())
