from datetime import timedelta
from pathlib import Path
import logging
import requests
import requests_cache
from typing import Any, Dict, Optional, Union

class CacheService:
    """Service for managing API response caching"""
    
    def __init__(self, cache_name: str = 'newsy_cache', expire_after: int = 86400):
        """
        Initialize the cache service.
        
        Args:
            cache_name: Name for the cache database
            expire_after: Cache expiration time in seconds (default: 24 hours)
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path('cache')
        
        try:
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_path = self.cache_dir / f'{cache_name}.sqlite'
            self.expire_after = expire_after
            
            # Initialize the cache
            self.session = requests_cache.CachedSession(
                str(self.cache_path),
                expire_after=timedelta(seconds=expire_after),
                backend='sqlite',
                ignored_parameters=['api_key'],  # Don't cache the API key
                old_data_on_error=True  # Use cached data if there's an error
            )
            self.logger.info(f"Cache initialized at {self.cache_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            # Fall back to a non-caching session
            self.session = requests.Session()
    
    def get_cached_response(self, url: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Get a cached response if available, otherwise make a new request.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            
        Returns:
            Response data as a dictionary, or None if request fails
        """
        try:
            self.logger.debug(f"Fetching from cache: {url} with params: {params}")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Log cache status
            if hasattr(response, 'from_cache') and response.from_cache:
                self.logger.debug(f"Cache HIT for {url}")
            else:
                self.logger.debug(f"Cache MISS for {url}")
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {str(e)}", exc_info=True)
            return None
            
        except ValueError as e:
            self.logger.error(f"Failed to parse JSON response from {url}: {str(e)}")
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error in get_cached_response: {str(e)}", exc_info=True)
            return None
    
    def clear_cache(self) -> bool:
        """
        Clear the entire cache.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if hasattr(self.session, 'cache') and self.session.cache is not None:
                self.session.cache.clear()
                self.logger.info("Cache cleared successfully")
                return True
            else:
                self.logger.warning("Cannot clear cache: No cache available")
                return False
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}", exc_info=True)
            return False
        
    def remove_expired_responses(self) -> bool:
        """
        Remove expired responses from the cache.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if hasattr(self.session, 'remove_expired_responses'):
                self.session.remove_expired_responses()
                self.logger.debug("Removed expired responses from cache")
                return True
            else:
                self.logger.warning("Cache does not support removing expired responses")
                return False
        except Exception as e:
            self.logger.error(f"Failed to remove expired responses: {str(e)}", exc_info=True)
            return False
        
    def get(self, key: str) -> Any:
        """
        Backward compatible get method that returns None.
        
        Note: This is a dummy implementation to maintain compatibility.
        The actual caching is handled by the requests-cache session.
        
        Args:
            key: Cache key (not used in this implementation)
            
        Returns:
            Always returns None
        """
        self.logger.warning("CacheService.get() is deprecated - using direct session caching")
        return None
