import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from serpapi import GoogleSearch
from dotenv import load_dotenv
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SerpAPIService:
    """Service for interacting with the SerpAPI for news search."""
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the SerpAPI service.
        
        Args:
            use_cache: Whether to use caching for API responses
        """
        self.api_key = os.getenv("SERPAPI_KEY")
        if not self.api_key:
            logger.warning("SERPAPI_KEY not found in environment variables")
            self.api_key = ""  # Will cause immediate failure on API call
            
        self.use_cache = use_cache
        self.cache = None
        
        if use_cache:
            try:
                from services.cache_service import CacheService
                self.cache = CacheService(cache_name='serpapi_cache', expire_after=86400)  # 24h cache
            except ImportError:
                logger.warning("CacheService not available, running without caching")

    def search_news(self, query: str, region: str = "us", max_results: int = 10, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Search for news articles using SerpAPI.
        
        Args:
            query: Search query string
            region: Region for search (us, global)
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results
            
        Returns:
            List of news article dictionaries
        """
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_filter = f"after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
            
            logger.info(f"Searching SerpAPI for: {query} in {region}")
            
            # Build search parameters based on SerpAPI documentation
            params = {
                "api_key": self.api_key,
                "engine": "google_news",  # Use google_news engine as per documentation
                "q": f"{query} {date_filter}",
                "hl": "en",
                "num": min(max(1, max_results), 100)  # Limit results between 1 and 100
            }
            
            # Add region parameter
            if region.lower() == "us":
                params["gl"] = "us"  # Use gl parameter for geolocation
            elif region.lower() == "global":
                # No specific parameter needed for global
                pass
            
            logger.debug(f"SerpAPI request params: { {k: v for k, v in params.items() if k != 'api_key'} }")
            
            # Execute search using the cached session if available
            if use_cache and self.cache:
                logger.info("Using cached session for request")
                logger.debug(f"Request URL: https://serpapi.com/search")
                
                response = self.cache.session.get("https://serpapi.com/search", params=params)
                logger.info(f"Response status: {response.status_code}")
                response.raise_for_status()
                
                results = response.json()
                logger.debug(f"Raw API response keys: {list(results.keys()) if isinstance(results, dict) else []}")
                
                if 'error' in results:
                    logger.error(f"SerpAPI error: {results.get('error')}")
                    return []
                    
            else:
                logger.info("Using direct SerpAPI request (no cache)")
                
                search = GoogleSearch(params)
                results = search.get_dict()
                logger.debug(f"Raw API response keys: {list(results.keys()) if isinstance(results, dict) else []}")
                
                if 'error' in results:
                    logger.error(f"SerpAPI error: {results.get('error')}")
                    return []
            
            # Dump the entire response for debugging (limited to avoid huge logs)
            response_str = str(results)[:500] if results else "Empty response"
            logger.debug(f"Raw response (truncated): {response_str}...")
            
            # Process results - check different possible response formats
            news_results = []
            
            if "news_results" in results:
                news_results = results.get("news_results", [])
                logger.info(f"Found {len(news_results)} news_results")
            elif "organic_results" in results:
                news_results = results.get("organic_results", [])
                logger.info(f"Found {len(news_results)} organic_results")
            else:
                logger.warning("No news_results or organic_results found in response")
                logger.debug(f"Available keys in response: {list(results.keys()) if isinstance(results, dict) else []}")
            
            processed = self._process_news_results(news_results)
            logger.info(f"Processed {len(processed)} news articles")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error searching news with SerpAPI: {str(e)}")
            return []
    
    def _process_news_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process raw news results from SerpAPI.
        
        Args:
            results: Raw results from SerpAPI
            
        Returns:
            List of processed article dictionaries
        """
        if not results:
            logger.warning("No results to process in _process_news_results")
            return []
        
        # Log first result to understand structure
        if results and len(results) > 0:
            logger.debug(f"First result structure: {results[0]}")
            logger.debug(f"First result keys: {list(results[0].keys()) if isinstance(results[0], dict) else 'Not a dict'}")
            
        processed = []
        for item in results:
            try:
                # Skip non-dict items
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item: {type(item)}: {item}")
                    continue
                    
                # Handle different possible field names
                title = item.get("title", "").strip()
                
                # Source can be a dictionary with a name field or a string
                source = ""
                if isinstance(item.get("source"), dict):
                    source = item.get("source", {}).get("name", "").strip()
                else:
                    source = str(item.get("source", "")).strip()
                
                # URL is in the link field in SerpAPI responses
                url = item.get("link", "").strip()
                if not url:
                    # Fallback to url field if link is not present
                    url = item.get("url", "").strip()
                
                # Snippet is in the snippet field in SerpAPI responses
                snippet = item.get("snippet", "").strip()
                if not snippet:
                    # Fallback to description field if snippet is not present
                    snippet = item.get("description", "").strip()
                
                # Date is a relative string in SerpAPI responses (e.g., "15 hours ago")
                date = item.get("date", "").strip()
                if not date:
                    date = item.get("published_date", "").strip()
                
                # Create the processed item with all fields
                processed_item = {
                    "title": title,
                    "source": source,
                    "date": date,
                    "url": url,
                    "snippet": snippet,
                }
                
                # Add thumbnail if available
                if "thumbnail" in item and item["thumbnail"]:
                    processed_item["thumbnail"] = item["thumbnail"].strip()
                
                # Add position if available
                if "position" in item:
                    processed_item["position"] = item["position"]
                
                # Add favicon if available
                if "favicon" in item and item["favicon"]:
                    processed_item["favicon"] = item["favicon"].strip()
                
                processed.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing news item: {e}")
                continue
                
        logger.info(f"Processed {len(processed)} news items from {len(results)} results")
        return processed
        
    def clear_cache(self) -> bool:
        """Clear the cache if caching is enabled."""
        if self.cache:
            self.cache.clear()
            return True
        return False
