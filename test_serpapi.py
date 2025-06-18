import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)
sys.path.append(str(Path(project_root) / "src"))
sys.path.append(str(Path(project_root) / "mcp" / "v1" / "src"))

# Check if SerpAPI key is loaded
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
print(f"SerpAPI Key: {'*' * 8 + SERPAPI_KEY[-4:] if SERPAPI_KEY else 'Not found'}")

# Initialize SerpAPI service
try:
    from services.serpapi_service import SerpAPIService
    
    print("\nTesting SerpAPI service...")
    serp = SerpAPIService(use_cache=False)
    
    # Test search with a simple query
    print("\nSearching for 'technology' news...")
    results = serp.search_news(
        query="technology",
        region="us",
        max_results=3,
        use_cache=False
    )
    
    print(f"\nFound {len(results)} results:")
    
    # Print the raw structure of the first result
    if results:
        print("\nFirst result raw structure:")
        import json
        print(json.dumps(results[0], indent=2))
    
    # Print the first 3 results in a more readable format
    for i, result in enumerate(results[:3], 1):
        print(f"\n--- Result {i} ---")
        for key, value in result.items():
            print(f"{key}: {value}")
        
        # Also print specific fields we're interested in
        print(f"\nTitle: {result.get('title', 'N/A')}")
        print(f"Source: {result.get('source', 'N/A')}")  # Source is already a string
        print(f"URL: {result.get('url', 'N/A')}")  # URL is stored in 'url' field
        print(f"Snippet: {result.get('snippet', 'N/A')}")
        
except ImportError as e:
    print(f"Error importing SerpAPIService: {e}")
except Exception as e:
    print(f"Error testing SerpAPI: {e}")
    raise
