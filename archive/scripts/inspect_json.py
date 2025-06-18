import json
import sys
from pathlib import Path

# Find the most recent serpapi_results file
result_files = list(Path('.').glob('serpapi_results_*.json'))
if not result_files:
    print("No SerpAPI result files found.")
    sys.exit(1)

# Sort by modification time (newest first)
latest_file = sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
print(f"Inspecting {latest_file}...")

try:
    # Load the JSON data
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Check for key structures
    print(f"News results present: {'news_results' in data}")
    print(f"Organic results present: {'organic_results' in data}")
    
    # If news_results exists, show details
    if 'news_results' in data:
        news = data['news_results']
        print(f"Number of news results: {len(news)}")
        
        if news:
            first = news[0]
            print(f"First news item keys: {list(first.keys())}")
            
            # Check source structure
            source = first.get('source')
            print(f"Source type: {type(source).__name__}")
            if isinstance(source, dict):
                print(f"Source keys: {list(source.keys())}")
            else:
                print(f"Source value: {source}")
                
            # Check link structure
            print(f"Link: {first.get('link', 'N/A')}")
            print(f"Title: {first.get('title', 'N/A')}")
    
    # Check for alternative result structures
    if 'organic_results' in data:
        organic = data['organic_results']
        print(f"\nNumber of organic results: {len(organic)}")
        
        if organic:
            first = organic[0]
            print(f"First organic item keys: {list(first.keys())}")
    
    # Check search parameters
    if 'search_parameters' in data:
        params = data['search_parameters']
        print(f"\nSearch parameters: {params}")
    
except Exception as e:
    print(f"Error inspecting JSON: {e}")
    import traceback
    traceback.print_exc()
