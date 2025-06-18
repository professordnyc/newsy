#!/usr/bin/env python
"""
Test script for the /search/headlines/classified endpoint.
This script sends a request to the MCP server and displays the classified results.
"""

import json
import requests
import argparse
from pprint import pprint

def test_classified_endpoint(query="technology", region="us", max_results=3, use_cache=True):
    """
    Test the /search/headlines/classified endpoint with the given parameters.
    
    Args:
        query: Search query string
        region: Region code (e.g., 'us', 'gb')
        max_results: Maximum number of results to return
        use_cache: Whether to use cached results if available
    
    Returns:
        The response from the endpoint
    """
    url = "http://localhost:8000/search/headlines/classified"
    
    payload = {
        "query": query,
        "region": region,
        "max_results": max_results,
        "use_cache": use_cache
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending request to {url} with payload: {payload}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for non-200 responses
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            try:
                print(f"Response content: {e.response.json()}")
            except:
                print(f"Response content: {e.response.text}")
        return None

def display_results(results):
    """
    Display the results in a readable format.
    
    Args:
        results: List of NewsArticle objects from the API
    """
    if not results:
        print("No results returned.")
        return
    
    print(f"\n=== Found {len(results)} classified articles ===\n")
    
    for i, article in enumerate(results):
        print(f"Article {i+1}:")
        print(f"  Title: {article.get('title', 'N/A')}")
        print(f"  Source: {article.get('source', 'N/A')}")
        print(f"  Classification: {article.get('classification', 'unknown')}")
        
        # Get confidence if available
        confidence = "N/A"
        if article.get('metadata') and 'classification_confidence' in article['metadata']:
            confidence = f"{article['metadata']['classification_confidence']:.2f}"
        
        print(f"  Confidence: {confidence}")
        print(f"  URL: {article.get('url', 'N/A')}")
        print(f"  Snippet: {article.get('snippet', 'N/A')[:100]}...")
        print()
    
    # Save results to file for further analysis
    with open("classified_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to classified_results.json")

def main():
    parser = argparse.ArgumentParser(description="Test the /search/headlines/classified endpoint")
    parser.add_argument("--query", default="technology", help="Search query")
    parser.add_argument("--region", default="us", help="Region code (e.g., 'us', 'gb')")
    parser.add_argument("--max-results", type=int, default=3, help="Maximum number of results")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    
    args = parser.parse_args()
    
    results = test_classified_endpoint(
        query=args.query,
        region=args.region,
        max_results=args.max_results,
        use_cache=not args.no_cache
    )
    
    if results:
        display_results(results)

if __name__ == "__main__":
    main()
