import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

from src.services.cache_service import CacheService

def test_cache_service():
    """Test the CacheService functionality."""
    print("\n=== Testing CacheService ===")
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize cache service
    try:
        logger.info("Initializing CacheService...")
        cache = CacheService(cache_name='test_cache', expire_after=60)  # 1 minute cache
        logger.info("✓ CacheService initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize CacheService: {e}")
        return False
    
    # Test URL
    test_url = "https://jsonplaceholder.typicode.com/todos/1"
    
    try:
        # First request (should be a miss)
        logger.info("\n--- First request (should be a MISS) ---")
        result = cache.get_cached_response(test_url)
        logger.info(f"Response: {result}")
        
        if not result:
            logger.error("✗ First request failed")
            return False
        
        # Second request (should be a hit)
        logger.info("\n--- Second request (should be a HIT) ---")
        result = cache.get_cached_response(test_url)
        logger.info(f"Response: {result}")
        
        if not result:
            logger.error("✗ Second request failed")
            return False
        
        # Test clearing cache
        logger.info("\n--- Testing cache clear ---")
        if cache.clear_cache():
            logger.info("✓ Cache cleared successfully")
            
            # Verify cache is cleared
            logger.info("\n--- Third request after clear (should be a MISS) ---")
            result = cache.get_cached_response(test_url)
            if result:
                logger.info("✓ Cache clear verified (got new response)")
            else:
                logger.error("✗ Cache clear verification failed")
                return False
        else:
            logger.error("✗ Failed to clear cache")
            return False
            
        logger.info("\n=== All tests passed successfully! ===")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_cache_service()
