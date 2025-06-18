import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project directory to the Python path
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

# Import the SerpAPIService
from src.services.serpapi_service import SerpAPIService

def main():
    print("Testing SerpAPI Integration")
    
    # Initialize the SerpAPI service
    serp_service = SerpAPIService(use_cache=True)
    print(f"SerpAPI service initialized with API key: {serp_service.api_key[:4]}****")
    
    # Test the search_news method
    query = "technology news"
    region = "us"
    max_results = 5
    
    print(f"\nSearching for '{query}' in {region}...")
    results = serp_service.search_news(
        query=query,
        region=region,
        max_results=max_results,
        use_cache=True
    )
    
    # Print the results
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Title: {result.get('title', 'No title')}")
        print(f"  Source: {result.get('source', 'Unknown')}")
        print(f"  URL: {result.get('url', 'No URL')}")
        print(f"  Date: {result.get('date', 'No date')}")
        print(f"  Snippet: {result.get('snippet', 'No snippet')[:100]}...")
    
    # Save the results to a file for further inspection
    with open("serpapi_integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to serpapi_integration_test_results.json")

if __name__ == "__main__":
    main()
