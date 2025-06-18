import os
import json
import sys
import pprint
from pathlib import Path

# Output file for analysis results
output_file = "serpapi_analysis_results.txt"

# Find the most recent serpapi_results file
result_files = list(Path('.').glob('serpapi_results_*.json'))
if not result_files:
    print("No SerpAPI result files found.")
    sys.exit(1)

# Sort by modification time (newest first)
latest_file = sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
print(f"Analyzing {latest_file}...")
print(f"Results will be written to {output_file}")

# Open output file
with open(output_file, 'w') as out:
    try:
        # Load the JSON data
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Write available top-level keys
        out.write(f"Top-level keys in response: {list(data.keys())}\n\n")
        
        # Check for news_results
        if 'news_results' in data:
            news = data['news_results']
            out.write(f"Found {len(news)} news results\n")
            
            # Write details of first 3 results
            for i, item in enumerate(news[:3]):
                out.write(f"\n--- Result {i+1} ---\n")
                out.write(f"Keys in result: {list(item.keys())}\n")
                out.write(f"Title: {item.get('title', 'N/A')}\n")
                
                # Handle source which could be string or dict
                source = item.get('source', 'N/A')
                if isinstance(source, dict):
                    out.write(f"Source (dict): {source}\n")
                else:
                    out.write(f"Source (string): {source}\n")
                
                out.write(f"Link: {item.get('link', 'N/A')}\n")
                out.write(f"Date: {item.get('date', 'N/A')}\n")
                
        # Check for organic_results (alternative location)
        elif 'organic_results' in data:
            results = data['organic_results']
            out.write(f"\nFound {len(results)} organic results\n")
            
            # Write details of first 3 results
            for i, item in enumerate(results[:3]):
                out.write(f"\n--- Organic Result {i+1} ---\n")
                out.write(f"Keys in result: {list(item.keys())}\n")
                
        else:
            out.write("\nNo news_results or organic_results found.\n")
            out.write("Checking other potential result arrays...\n")
            
            # Look for any arrays in the response that might contain news
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    out.write(f"\nFound array in '{key}' with {len(value)} items\n")
                    if value and isinstance(value[0], dict):
                        out.write(f"First item keys: {list(value[0].keys())}\n")
                        
                        # If it has title/link, it's probably a news result
                        if 'title' in value[0] or 'link' in value[0]:
                            out.write(f"\nPossible news results in '{key}':\n")
                            for i, item in enumerate(value[:2]):
                                out.write(f"\n--- Item {i+1} ---\n")
                                for k, v in item.items():
                                    if isinstance(v, (str, int, float, bool)) or v is None:
                                        out.write(f"{k}: {v}\n")
                                    else:
                                        out.write(f"{k}: {type(v)}\n")
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
