#!/usr/bin/env python3
"""
Perplexity API Key Management and Rotation
Handles multiple API keys with automatic rotation and health monitoring
"""

import os
import time
import json
import logging
import requests
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class KeyStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    INVALID = "invalid"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

@dataclass
class APIKeyInfo:
    """API key information and status"""
    key_id: str
    key_hash: str  # First 8 chars for identification
    status: KeyStatus
    last_used: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    error_count: int = 0
    rate_limit_reset: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    daily_quota_used: int = 0
    daily_quota_limit: int = 1000  # Default Perplexity limit

@dataclass
class APIResponse:
    """Structured API response"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    key_used: Optional[str] = None
    rate_limit_info: Optional[Dict[str, Any]] = None

class PerplexityKeyManager:
    """
    Manages multiple Perplexity API keys with intelligent rotation
    """
    
    def __init__(self, config_file: str = "config/pplx_keys.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)
        
        # API configuration
        self.base_url = "https://api.perplexity.ai"
        self.default_model = "llama-3.1-sonar-small-128k-online"
        self.request_timeout = 30
        
        # Rate limiting
        self.requests_per_minute = 20  # Conservative limit
        self.daily_quota_limit = 1000  # Default limit
        
        # Key management
        self.keys: Dict[str, APIKeyInfo] = {}
        self.current_key_index = 0
        self.rotation_lock = threading.Lock()
        
        # Load keys from environment and config
        self._load_keys()
        
        # Validation on startup
        self._validate_all_keys()
    
    def _load_keys(self):
        """Load API keys from environment variables and config file"""
        # Load from environment variables
        env_keys = []
        for i in range(1, 21):  # Support up to 20 keys
            key_var = f"PPLX_API_KEY_{i}"
            if key_var in os.environ:
                key_value = os.environ[key_var].strip()
                if key_value:
                    env_keys.append(key_value)
        
        # Also check for single key
        if "PPLX_API_KEY" in os.environ:
            single_key = os.environ["PPLX_API_KEY"].strip()
            if single_key and single_key not in env_keys:
                env_keys.append(single_key)
        
        # Load from config file if exists
        config_keys = []
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    config_keys = config_data.get('api_keys', [])
            except Exception as e:
                logger.warning(f"Failed to load key config file: {e}")
        
        # Combine and deduplicate
        all_keys = list(set(env_keys + config_keys))
        
        if not all_keys:
            logger.warning("No Perplexity API keys found in environment or config")
            return
        
        # Initialize key info
        for i, key in enumerate(all_keys):
            key_id = f"key_{i+1}"
            key_hash = key[:8] + "..." if len(key) > 8 else key
            
            self.keys[key] = APIKeyInfo(
                key_id=key_id,
                key_hash=key_hash,
                status=KeyStatus.UNKNOWN
            )
        
        logger.info(f"Loaded {len(self.keys)} Perplexity API keys")
    
    def _validate_all_keys(self):
        """Validate all API keys by testing them"""
        if not self.keys:
            logger.error("No API keys to validate")
            return
        
        logger.info("Validating Perplexity API keys...")
        
        for key, key_info in self.keys.items():
            try:
                # Test the key with a simple request
                response = self._test_key(key)
                
                if response.success:
                    key_info.status = KeyStatus.ACTIVE
                    key_info.last_success = datetime.now()
                    key_info.error_count = 0
                    logger.info(f"Key {key_info.key_id} ({key_info.key_hash}): Valid")
                else:
                    if response.status_code == 401:
                        key_info.status = KeyStatus.INVALID
                        logger.error(f"Key {key_info.key_id} ({key_info.key_hash}): Invalid")
                    elif response.status_code == 429:
                        key_info.status = KeyStatus.RATE_LIMITED
                        logger.warning(f"Key {key_info.key_id} ({key_info.key_hash}): Rate limited")
                    else:
                        key_info.status = KeyStatus.UNKNOWN
                        logger.warning(f"Key {key_info.key_id} ({key_info.key_hash}): Unknown status ({response.status_code})")
                
            except Exception as e:
                key_info.status = KeyStatus.UNKNOWN
                key_info.last_error = datetime.now()
                key_info.error_count += 1
                logger.error(f"Key {key_info.key_id} validation failed: {e}")
        
        # Update current key index to first active key
        self._rotate_to_next_active_key()
        
        # Save key status
        self._save_key_status()
    
    def _test_key(self, api_key: str) -> APIResponse:
        """Test an API key with a minimal request"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Minimal test payload
        test_payload = {
            "model": self.default_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Test"
                }
            ],
            "max_tokens": 1,
            "temperature": 0
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=self.request_timeout
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Parse rate limit headers
            rate_limit_info = {
                'remaining': response.headers.get('x-ratelimit-remaining'),
                'reset': response.headers.get('x-ratelimit-reset'),
                'limit': response.headers.get('x-ratelimit-limit')
            }
            
            if response.status_code == 200:
                return APIResponse(
                    success=True,
                    status_code=response.status_code,
                    response_time_ms=response_time,
                    rate_limit_info=rate_limit_info
                )
            else:
                return APIResponse(
                    success=False,
                    status_code=response.status_code,
                    error=response.text,
                    response_time_ms=response_time,
                    rate_limit_info=rate_limit_info
                )
                
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                error="Request timeout",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except requests.exceptions.RequestException as e:
            return APIResponse(
                success=False,
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_active_key(self) -> Optional[str]:
        """Get the current active API key"""
        if not self.keys:
            return None
        
        active_keys = [key for key, info in self.keys.items() if info.status == KeyStatus.ACTIVE]
        
        if not active_keys:
            logger.warning("No active API keys available")
            return None
        
        # Return current key if still active
        if self.current_key_index < len(active_keys):
            return active_keys[self.current_key_index]
        
        # Reset to first active key
        self.current_key_index = 0
        return active_keys[0]
    
    def _rotate_to_next_active_key(self):
        """Rotate to the next active API key"""
        with self.rotation_lock:
            active_keys = [key for key, info in self.keys.items() if info.status == KeyStatus.ACTIVE]
            
            if not active_keys:
                logger.error("No active API keys available for rotation")
                return
            
            self.current_key_index = (self.current_key_index + 1) % len(active_keys)
            new_key = active_keys[self.current_key_index]
            key_info = self.keys[new_key]
            
            logger.info(f"Rotated to key {key_info.key_id} ({key_info.key_hash})")
    
    def make_request(self, messages: List[Dict[str, str]], **kwargs) -> APIResponse:
        """
        Make a request to Perplexity API with automatic key rotation
        """
        max_retries = len([k for k in self.keys.values() if k.status == KeyStatus.ACTIVE])
        
        for attempt in range(max_retries):
            api_key = self.get_active_key()
            
            if not api_key:
                return APIResponse(
                    success=False,
                    error="No active API keys available"
                )
            
            key_info = self.keys[api_key]
            
            # Check daily quota
            if key_info.daily_quota_used >= key_info.daily_quota_limit:
                logger.warning(f"Key {key_info.key_id} has reached daily quota")
                key_info.status = KeyStatus.RATE_LIMITED
                self._rotate_to_next_active_key()
                continue
            
            # Make the request
            response = self._make_api_request(api_key, messages, **kwargs)
            
            # Update key statistics
            key_info.last_used = datetime.now()
            key_info.total_requests += 1
            
            if response.success:
                key_info.last_success = datetime.now()
                key_info.successful_requests += 1
                key_info.daily_quota_used += 1
                key_info.error_count = 0  # Reset error count on success
                response.key_used = key_info.key_id
                return response
            
            else:
                key_info.last_error = datetime.now()
                key_info.error_count += 1
                
                # Handle specific error types
                if response.status_code == 401:
                    logger.error(f"Key {key_info.key_id} is invalid")
                    key_info.status = KeyStatus.INVALID
                    self._rotate_to_next_active_key()
                    continue
                
                elif response.status_code == 429:
                    logger.warning(f"Key {key_info.key_id} is rate limited")
                    key_info.status = KeyStatus.RATE_LIMITED
                    
                    # Parse rate limit reset time
                    if response.rate_limit_info and response.rate_limit_info.get('reset'):
                        try:
                            reset_time = datetime.fromtimestamp(
                                int(response.rate_limit_info['reset'])
                            )
                            key_info.rate_limit_reset = reset_time
                        except:
                            pass
                    
                    self._rotate_to_next_active_key()
                    continue
                
                # For other errors, try next key if this one has too many errors
                elif key_info.error_count >= 3:
                    logger.warning(f"Key {key_info.key_id} has too many errors, rotating")
                    self._rotate_to_next_active_key()
                    continue
                
                # Return the error if it's not a key-specific issue
                return response
        
        return APIResponse(
            success=False,
            error="All API keys exhausted"
        )
    
    def _make_api_request(self, api_key: str, messages: List[Dict[str, str]], **kwargs) -> APIResponse:
        """Make the actual API request"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Build payload
        payload = {
            "model": kwargs.get("model", self.default_model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        # Add optional parameters
        for param in ["top_p", "top_k", "presence_penalty", "frequency_penalty"]:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Parse rate limit headers
            rate_limit_info = {
                'remaining': response.headers.get('x-ratelimit-remaining'),
                'reset': response.headers.get('x-ratelimit-reset'),
                'limit': response.headers.get('x-ratelimit-limit')
            }
            
            if response.status_code == 200:
                data = response.json()
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status_code,
                    response_time_ms=response_time,
                    rate_limit_info=rate_limit_info
                )
            else:
                return APIResponse(
                    success=False,
                    status_code=response.status_code,
                    error=response.text,
                    response_time_ms=response_time,
                    rate_limit_info=rate_limit_info
                )
                
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                error="Request timeout",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except requests.exceptions.RequestException as e:
            return APIResponse(
                success=False,
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_sentiment_analysis(self, text: str) -> APIResponse:
        """Get sentiment analysis for text"""
        messages = [
            {
                "role": "system",
                "content": "You are a financial sentiment analysis expert. Analyze the given text and return only a JSON object with 'sentiment' (positive/negative/neutral) and 'confidence' (0.0-1.0) fields."
            },
            {
                "role": "user",
                "content": f"Analyze the sentiment of this financial text: {text}"
            }
        ]
        
        return self.make_request(
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
    
    def get_narrative_summary(self, headlines: List[str]) -> APIResponse:
        """Get narrative summary from headlines"""
        headlines_text = "\n".join(f"- {headline}" for headline in headlines)
        
        messages = [
            {
                "role": "system",
                "content": "You are a crypto market analyst. Summarize the overall market narrative from these headlines. Return only a JSON object with 'narrative' (string), 'sentiment' (positive/negative/neutral), and 'confidence' (0.0-1.0) fields."
            },
            {
                "role": "user",
                "content": f"Summarize the market narrative from these headlines:\n{headlines_text}"
            }
        ]
        
        return self.make_request(
            messages=messages,
            max_tokens=200,
            temperature=0.3
        )
    
    def get_key_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all API keys"""
        status = {}
        
        for key, info in self.keys.items():
            status[info.key_id] = {
                'key_hash': info.key_hash,
                'status': info.status.value,
                'last_used': info.last_used.isoformat() if info.last_used else None,
                'last_success': info.last_success.isoformat() if info.last_success else None,
                'total_requests': info.total_requests,
                'successful_requests': info.successful_requests,
                'error_count': info.error_count,
                'daily_quota_used': info.daily_quota_used,
                'daily_quota_limit': info.daily_quota_limit,
                'success_rate': info.successful_requests / max(info.total_requests, 1)
            }
        
        return status
    
    def _save_key_status(self):
        """Save key status to config file"""
        try:
            status_data = {
                'last_updated': datetime.now().isoformat(),
                'keys': self.get_key_status()
            }
            
            status_file = self.config_file.parent / "pplx_key_status.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save key status: {e}")
    
    def reset_daily_quotas(self):
        """Reset daily quotas for all keys (call daily)"""
        for key_info in self.keys.values():
            key_info.daily_quota_used = 0
        
        logger.info("Daily quotas reset for all keys")
        self._save_key_status()
    
    def refresh_rate_limited_keys(self):
        """Check if rate-limited keys can be reactivated"""
        now = datetime.now()
        
        for key_info in self.keys.values():
            if key_info.status == KeyStatus.RATE_LIMITED:
                # Check if rate limit has expired
                if key_info.rate_limit_reset and now > key_info.rate_limit_reset:
                    key_info.status = KeyStatus.ACTIVE
                    key_info.rate_limit_reset = None
                    logger.info(f"Key {key_info.key_id} reactivated after rate limit")
        
        self._save_key_status()

# Global key manager instance
pplx_manager = PerplexityKeyManager()

def get_sentiment_analysis(text: str) -> APIResponse:
    """Get sentiment analysis for text"""
    return pplx_manager.get_sentiment_analysis(text)

def get_narrative_summary(headlines: List[str]) -> APIResponse:
    """Get narrative summary from headlines"""
    return pplx_manager.get_narrative_summary(headlines)

def get_pplx_key_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all Perplexity API keys"""
    return pplx_manager.get_key_status()

if __name__ == "__main__":
    # Test the Perplexity key management system
    print("🔑 Testing Perplexity API Key Management")
    print("=" * 45)
    
    # Show key status
    status = get_pplx_key_status()
    print(f"\nLoaded {len(status)} API keys:")
    
    for key_id, info in status.items():
        status_icon = {
            "active": "✅",
            "rate_limited": "⏳",
            "invalid": "❌",
            "expired": "⏰",
            "unknown": "❓"
        }.get(info['status'], "❓")
        
        print(f"  {status_icon} {key_id} ({info['key_hash']}): {info['status']}")
        print(f"     Requests: {info['total_requests']} | Success Rate: {info['success_rate']:.1%}")
        
        if info['last_success']:
            last_success = datetime.fromisoformat(info['last_success'])
            print(f"     Last Success: {last_success.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test sentiment analysis
    print("\nTesting sentiment analysis...")
    test_text = "Bitcoin reaches new all-time high as institutional adoption accelerates"
    
    response = get_sentiment_analysis(test_text)
    
    if response.success:
        print(f"✅ Sentiment analysis successful")
        print(f"   Response time: {response.response_time_ms:.1f}ms")
        print(f"   Key used: {response.key_used}")
        if response.data:
            print(f"   Result: {response.data}")
    else:
        print(f"❌ Sentiment analysis failed: {response.error}")
    
    print("\n✅ Perplexity API key management ready!")
