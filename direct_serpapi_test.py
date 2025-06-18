import os
import json
import datetime
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("SERPAPI_KEY")
if not api_key:
    print("Error: SERPAPI_KEY not found in environment variables")
    exit(1)

print(f"Using SerpAPI key: {'*' * 8 + api_key[-4:] if len(api_key) > 4 else '****'}")

# Set up the search parameters
params = {
    "q": "technology news",
    "tbm": "nws",  # News search
    "api_key": api_key,
    "gl": "us"     # Location parameter for US results
}

# Create a timestamp for the output file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"serpapi_results_{timestamp}.json"

try:
    print("\nMaking direct request to SerpAPI...")
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Check for errors
    if "error" in results:
        print(f"SerpAPI error: {results['error']}")
        with open(output_file, 'w') as f:
            json.dump({"error": results['error']}, f, indent=2)
        print(f"Error details saved to {output_file}")
        exit(1)
    
    # Save full results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to {output_file}")
    print(f"Response keys: {list(results.keys())}")
    
    # Check if news_results exists
    if "news_results" in results:
        news = results["news_results"]
        print(f"\nFound {len(news)} news results")
        
        # Print summary of first few results
        for i, item in enumerate(news[:3]):
            print(f"\nResult {i+1}:")
            print(f"Title: {item.get('title', 'N/A')}")
            
            # Handle source field which could be a string or dict
            source = item.get('source', 'N/A')
            if isinstance(source, dict):
                source_name = source.get('name', 'N/A')
                print(f"Source: {source_name}")
            else:
                print(f"Source: {source}")
                
            print(f"Link: {item.get('link', 'N/A')}")
    else:
        print("\nNo 'news_results' key found in response")
        print("Available keys:", list(results.keys()))
        
        # Try to find where the news results might be
        for key in results:
            if isinstance(results[key], list) and len(results[key]) > 0:
                print(f"\nPossible results in '{key}': {len(results[key])} items")
                if results[key]:
                    first_item = results[key][0]
                    if isinstance(first_item, dict):
                        print(f"First item keys: {list(first_item.keys())}")
                    else:
                        print(f"First item type: {type(first_item)}")
                        print(f"First item preview: {str(first_item)[:50]}...")
    
    print(f"\nCheck {output_file} for complete results")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
