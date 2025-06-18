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
print(f"Analyzing {latest_file}...")

# Output file
output_file = "serpapi_structure_analysis.txt"

try:
    # Load the JSON data
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Open output file
    with open(output_file, 'w') as out:
        # Write top-level structure
        out.write("=== SerpAPI Response Structure ===\n\n")
        out.write(f"Top-level keys: {list(data.keys())}\n\n")
        
        # Write search parameters
        if 'search_parameters' in data:
            out.write("=== Search Parameters ===\n")
            for k, v in data['search_parameters'].items():
                out.write(f"{k}: {v}\n")
            out.write("\n")
        
        # Check for news_results
        if 'news_results' in data:
            news = data['news_results']
            out.write(f"=== News Results ({len(news)} items) ===\n")
            
            # Write structure of first news item
            if news:
                first_item = news[0]
                out.write(f"News item structure (keys): {list(first_item.keys())}\n\n")
                
                # Write sample of first news item
                out.write("=== Sample News Item ===\n")
                for k, v in first_item.items():
                    if isinstance(v, dict):
                        out.write(f"{k} (dict): {v}\n")
                    elif isinstance(v, list):
                        out.write(f"{k} (list of {len(v)} items): {v[:50] if len(str(v)) > 50 else v}\n")
                    else:
                        out.write(f"{k}: {v}\n")
        else:
            out.write("No 'news_results' key found in response\n")
        
        # Check for organic_results as alternative
        if 'organic_results' in data:
            results = data['organic_results']
            out.write(f"\n=== Organic Results ({len(results)} items) ===\n")
            
            # Write structure of first organic item
            if results:
                first_item = results[0]
                out.write(f"Organic item structure (keys): {list(first_item.keys())}\n")
        
        # Look for any other potential result arrays
        out.write("\n=== Other Potential Result Arrays ===\n")
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0 and key not in ['news_results', 'organic_results']:
                out.write(f"{key}: {len(value)} items\n")
                if value and isinstance(value[0], dict):
                    out.write(f"  First item keys: {list(value[0].keys())}\n")
    
    print(f"Analysis saved to {output_file}")
    
except Exception as e:
    print(f"Error analyzing results: {e}")
    import traceback
    traceback.print_exc()
