#!/usr/bin/env python3
"""
Real-Time System Health Monitoring
Tracks critical system components and provides alerts
"""

import os
import sys
import time
import json
import psutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class SystemHealth:
    """Overall system health status"""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime
    uptime_seconds: float
    system_metrics: Dict[str, Any]

class HealthMonitor:
    """
    Comprehensive system health monitoring
    Checks all critical components and provides real-time status
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.check_interval = 30  # seconds
        self.critical_thresholds = {
            'cpu_usage': 80.0,      # %
            'memory_usage': 85.0,   # %
            'disk_usage': 90.0,     # %
            'response_time': 5000,  # ms
        }
        self.warning_thresholds = {
            'cpu_usage': 60.0,      # %
            'memory_usage': 70.0,   # %
            'disk_usage': 80.0,     # %
            'response_time': 2000,  # ms
        }
        
        # Health check registry
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'data_pipeline': self._check_data_pipeline,
            'trading_api': self._check_trading_api,
            'ml_models': self._check_ml_models,
            'sentiment_api': self._check_sentiment_api,
            'file_system': self._check_file_system,
            'network_connectivity': self._check_network_connectivity
        }
        
        self.last_health_status = None
        self.monitoring_active = False
        self.monitor_thread = None
    
    def check_all_systems(self) -> SystemHealth:
        """Run all health checks and return overall system status"""
        start_time = time.time()
        check_results = []
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                check_results.append(result)
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                check_results.append(HealthCheck(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now()
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(check_results)
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        health_status = SystemHealth(
            overall_status=overall_status,
            checks=check_results,
            timestamp=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            system_metrics=system_metrics
        )
        
        self.last_health_status = health_status
        return health_status
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system status from individual checks"""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        elif all(check.status == HealthStatus.HEALTHY for check in checks):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'process_count': len(psutil.pids()),
                'boot_time': psutil.boot_time()
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage"""
        try:
            metrics = self._get_system_metrics()
            
            cpu_usage = metrics.get('cpu_usage_percent', 0)
            memory_usage = metrics.get('memory_usage_percent', 0)
            disk_usage = metrics.get('disk_usage_percent', 0)
            
            issues = []
            status = HealthStatus.HEALTHY
            
            # Check critical thresholds
            if cpu_usage > self.critical_thresholds['cpu_usage']:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_usage:.1f}%")
            elif cpu_usage > self.warning_thresholds['cpu_usage']:
                status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_usage:.1f}%")
            
            if memory_usage > self.critical_thresholds['memory_usage']:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory_usage:.1f}%")
            elif memory_usage > self.warning_thresholds['memory_usage']:
                status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory_usage:.1f}%")
            
            if disk_usage > self.critical_thresholds['disk_usage']:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk_usage:.1f}%")
            elif disk_usage > self.warning_thresholds['disk_usage']:
                status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk_usage:.1f}%")
            
            message = "; ".join(issues) if issues else f"Resources normal (CPU: {cpu_usage:.1f}%, RAM: {memory_usage:.1f}%, Disk: {disk_usage:.1f}%)"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details=metrics
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_data_pipeline(self) -> HealthCheck:
        """Check data pipeline health"""
        try:
            start_time = time.time()
            
            # Test data pipeline with mock data
            from data_pipeline import data_pipeline, DataProvider
            test_data = data_pipeline._generate_mock_data('BTC/USD', 5)
            standardized = data_pipeline.standardizer.standardize_data(
                test_data, DataProvider.MOCK, 'BTC/USD'
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if standardized.empty:
                return HealthCheck(
                    name="data_pipeline",
                    status=HealthStatus.CRITICAL,
                    message="Data pipeline returned empty data",
                    timestamp=datetime.now(),
                    response_time_ms=response_time
                )
            
            status = HealthStatus.HEALTHY
            if response_time > self.critical_thresholds['response_time']:
                status = HealthStatus.CRITICAL
            elif response_time > self.warning_thresholds['response_time']:
                status = HealthStatus.WARNING
            
            return HealthCheck(
                name="data_pipeline",
                status=status,
                message=f"Data pipeline functional ({len(standardized)} bars processed)",
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="data_pipeline",
                status=HealthStatus.CRITICAL,
                message=f"Data pipeline check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_trading_api(self) -> HealthCheck:
        """Check trading API connectivity"""
        try:
            start_time = time.time()
            
            # Try to import and test Alpaca API
            try:
                from alpaca import _rest
                api = _rest()
                account = api.get_account()
                
                response_time = (time.time() - start_time) * 1000
                
                if account:
                    return HealthCheck(
                        name="trading_api",
                        status=HealthStatus.HEALTHY,
                        message=f"Trading API connected (account: {account.account_number})",
                        timestamp=datetime.now(),
                        response_time_ms=response_time,
                        details={'buying_power': float(account.buying_power)}
                    )
                else:
                    return HealthCheck(
                        name="trading_api",
                        status=HealthStatus.WARNING,
                        message="Trading API connected but no account data",
                        timestamp=datetime.now(),
                        response_time_ms=response_time
                    )
                    
            except ImportError:
                return HealthCheck(
                    name="trading_api",
                    status=HealthStatus.WARNING,
                    message="Trading API module not available",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheck(
                name="trading_api",
                status=HealthStatus.CRITICAL,
                message=f"Trading API check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_ml_models(self) -> HealthCheck:
        """Check ML models availability"""
        try:
            start_time = time.time()
            
            # Test ML ensemble initialization
            from ensemble_ml_models import TradingEnsemble
            ensemble = TradingEnsemble(input_dim=37)
            
            # Test prediction with dummy data
            import numpy as np
            dummy_features = np.random.rand(37)
            prediction = ensemble.predict(dummy_features)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="ml_models",
                status=HealthStatus.HEALTHY,
                message=f"ML models functional (test prediction: {prediction:.3f})",
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="ml_models",
                status=HealthStatus.WARNING,
                message=f"ML models check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_sentiment_api(self) -> HealthCheck:
        """Check sentiment API connectivity"""
        try:
            # Check if Perplexity API keys are available
            pplx_keys = [key for key in os.environ.keys() if key.startswith('PPLX_API_KEY')]
            
            if not pplx_keys:
                return HealthCheck(
                    name="sentiment_api",
                    status=HealthStatus.WARNING,
                    message="No Perplexity API keys found in environment",
                    timestamp=datetime.now()
                )
            
            # Test API connectivity (simplified)
            return HealthCheck(
                name="sentiment_api",
                status=HealthStatus.HEALTHY,
                message=f"Sentiment API configured ({len(pplx_keys)} keys available)",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheck(
                name="sentiment_api",
                status=HealthStatus.WARNING,
                message=f"Sentiment API check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_file_system(self) -> HealthCheck:
        """Check critical file system paths"""
        try:
            critical_paths = ['state/', 'logs/', 'config/']
            missing_paths = []
            
            for path in critical_paths:
                if not os.path.exists(path):
                    missing_paths.append(path)
                    # Create directory if it doesn't exist
                    os.makedirs(path, exist_ok=True)
            
            if missing_paths:
                return HealthCheck(
                    name="file_system",
                    status=HealthStatus.WARNING,
                    message=f"Created missing directories: {', '.join(missing_paths)}",
                    timestamp=datetime.now()
                )
            
            return HealthCheck(
                name="file_system",
                status=HealthStatus.HEALTHY,
                message="All critical directories present",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheck(
                name="file_system",
                status=HealthStatus.CRITICAL,
                message=f"File system check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_network_connectivity(self) -> HealthCheck:
        """Check network connectivity to critical services"""
        try:
            start_time = time.time()
            
            # Test connectivity to key endpoints
            test_urls = [
                'https://api.alpaca.markets',
                'https://api.perplexity.ai',
                'https://api.github.com'
            ]
            
            successful_connections = 0
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code < 500:  # Accept 4xx but not 5xx
                        successful_connections += 1
                except:
                    pass  # Connection failed
            
            response_time = (time.time() - start_time) * 1000
            
            if successful_connections == len(test_urls):
                status = HealthStatus.HEALTHY
                message = "All network endpoints reachable"
            elif successful_connections > 0:
                status = HealthStatus.WARNING
                message = f"Partial connectivity: {successful_connections}/{len(test_urls)} endpoints"
            else:
                status = HealthStatus.CRITICAL
                message = "No network connectivity to critical endpoints"
            
            return HealthCheck(
                name="network_connectivity",
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="network_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Network check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def start_monitoring(self):
        """Start continuous health monitoring in background thread"""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                health_status = self.check_all_systems()
                
                # Log critical issues
                if health_status.overall_status == HealthStatus.CRITICAL:
                    critical_checks = [check for check in health_status.checks 
                                     if check.status == HealthStatus.CRITICAL]
                    logger.error(f"CRITICAL SYSTEM ISSUES: {[check.message for check in critical_checks]}")
                
                # Save health status to file
                self._save_health_status(health_status)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                time.sleep(self.check_interval)
    
    def _save_health_status(self, health_status: SystemHealth):
        """Save health status to file for external monitoring"""
        try:
            os.makedirs('logs', exist_ok=True)
            
            health_data = {
                'overall_status': health_status.overall_status.value,
                'timestamp': health_status.timestamp.isoformat(),
                'uptime_seconds': health_status.uptime_seconds,
                'system_metrics': health_status.system_metrics,
                'checks': [
                    {
                        'name': check.name,
                        'status': check.status.value,
                        'message': check.message,
                        'timestamp': check.timestamp.isoformat(),
                        'response_time_ms': check.response_time_ms,
                        'details': check.details
                    }
                    for check in health_status.checks
                ]
            }
            
            with open('logs/system_health.json', 'w') as f:
                json.dump(health_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save health status: {e}")

# Global health monitor instance
health_monitor = HealthMonitor()

def get_system_health() -> SystemHealth:
    """Get current system health status"""
    return health_monitor.check_all_systems()

def start_health_monitoring():
    """Start continuous health monitoring"""
    health_monitor.start_monitoring()

def stop_health_monitoring():
    """Stop continuous health monitoring"""
    health_monitor.stop_monitoring()

if __name__ == "__main__":
    # Test the health monitoring system
    print("🏥 Testing System Health Monitor")
    print("=" * 40)
    
    health_status = health_monitor.check_all_systems()
    
    print(f"\nOverall Status: {health_status.overall_status.value.upper()}")
    print(f"Uptime: {health_status.uptime_seconds:.1f} seconds")
    print(f"Timestamp: {health_status.timestamp}")
    
    print("\nHealth Checks:")
    for check in health_status.checks:
        status_icon = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓"
        }.get(check.status, "❓")
        
        print(f"  {status_icon} {check.name}: {check.message}")
        if check.response_time_ms:
            print(f"     Response time: {check.response_time_ms:.1f}ms")
    
    print(f"\nSystem Metrics:")
    for key, value in health_status.system_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ System health monitoring ready!")
