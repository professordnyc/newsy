import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_article' not in st.session_state:
    st.session_state.selected_article = None

def search_news(query: str, region: str = "us", max_results: int = 10, 
                 use_cache: bool = True, batch_size: int = 4, timeout: int = 30) -> List[Dict]:
    """Search for news articles using the MCP server.
    
    Args:
        query: Search query
        region: Region code (e.g., 'us', 'uk')
        max_results: Maximum number of results to return
        use_cache: Whether to use cached results
        batch_size: Number of parallel workers for classification
        timeout: Timeout in seconds for API requests
    
    Returns:
        List of article dictionaries
    """
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/search/headlines/classified",
            json={
                "query": query,
                "region": region.lower(),
                "max_results": max_results,
                "use_cache": use_cache,
                "batch_size": batch_size,
                "timeout": timeout
            },
            timeout=timeout + 5  # Add a buffer to the request timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching for news: {str(e)}")
        return []

def display_article(article: Dict):
    """Display detailed view of a single article."""
    with st.container():
        st.markdown(f"### {article.get('title', 'No title')}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Source: {article.get('source', 'Unknown')} • {article.get('date', '')}")
            if 'snippet' in article:
                st.write(article['snippet'])
            
            if 'url' in article:
                st.markdown(f"[Read full article]({article['url']})")
        
        with col2:
            if 'classification' in article:
                # Handle both string and dictionary formats
                classification_raw = article['classification']
                
                # If classification is a string (from MCP server)
                if isinstance(classification_raw, str):
                    label = classification_raw.title()
                    # Look for confidence in metadata
                    confidence = article.get('metadata', {}).get('classification_confidence', 0.5)
                # If classification is a dictionary (from fallback)
                elif isinstance(classification_raw, dict):
                    label = classification_raw.get('prediction', 'Unknown').title()
                    confidence = classification_raw.get('confidence', 0.5)
                else:
                    label = str(classification_raw).title()
                    confidence = 0.5
                
                # Display with appropriate styling
                if label.lower() == 'leading':
                    st.success(f"🔍 {label} ({(confidence * 100):.1f}%)")
                elif label.lower() in ('under_reported', 'under reported'):
                    st.warning(f"🔍 Under-reported ({(confidence * 100):.1f}%)")
                else:
                    st.info(f"🔍 {label} ({(confidence * 100):.1f}%)")
                    
                # Debug info
                with st.expander("Classification Details"):
                    st.write({
                        "Raw": classification_raw,
                        "Label": label,
                        "Confidence": confidence,
                        "Metadata": article.get('metadata', {})
                    })
        
        st.divider()

def main():
    st.set_page_config(
        page_title="Newsy - Headlines Interrogation",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Newsy")
    st.subheader("Headlines Interrogation and Classification")
    
    # Sidebar with search and filters
    with st.sidebar:
        st.header("🔍 Search")
        query = st.text_input("Search for news", "")
        
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("Region", ["US", "Global"], index=0)
        with col2:
            max_results = st.slider("Max results", 5, 50, 10, 5)
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            use_cache = st.checkbox("Use cache", True, 
                                   help="Use cached results when available to improve performance")
            
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider("Batch size", 1, 8, 4, 1,
                                      help="Number of parallel workers for classification")
            with col2:
                timeout = st.slider("Timeout (sec)", 10, 60, 30, 5,
                                   help="Timeout for API requests in seconds")
        
        search_button = st.button("Search", type="primary", use_container_width=True)
        
        st.divider()
        st.markdown("### About")
        st.markdown("Newsy helps you discover and analyze news articles, identifying both leading stories and under-reported news.")
    
    # Main content area
    if search_button and query:
        with st.spinner("Searching for news..."):
            # Use the advanced parameters from the UI
            results = search_news(
                query=query, 
                region=region, 
                max_results=max_results,
                use_cache=use_cache,
                batch_size=batch_size,
                timeout=timeout
            )[:max_results]
            
            # Apply simple heuristic classification if missing
            total = len(results)
            for idx, art in enumerate(results):
                cls = art.get('classification')
                if not isinstance(cls, dict) or 'prediction' not in cls:
                    heuristic = 'leading' if idx < total * 0.3 else 'under_reported'
                    art['classification'] = {
                        'prediction': heuristic,
                        'confidence': 0.5
                    }
            st.session_state.search_results = results
    
    # Display search results or selected article
    if st.session_state.search_results:
        st.subheader("📰 Search Results")
        
        # Display filters
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Show all articles", True)
        with col2:
            if not show_all:
                show_under_reported = st.checkbox("Show under-reported only", True)
        
        # Filter and display articles
        for idx, article in enumerate(st.session_state.search_results):
            if not show_all:
                classification_raw = article.get('classification', {})
                
                # Check if article is under-reported based on classification format
                if isinstance(classification_raw, str):
                    is_under_reported = classification_raw.lower() == 'under_reported'
                elif isinstance(classification_raw, dict):
                    is_under_reported = classification_raw.get('prediction', '').lower() == 'under_reported'
                else:
                    is_under_reported = False
                    
                if show_under_reported and not is_under_reported:
                    continue
            
            if st.button(
                f"{article.get('title', 'No title')[:80]}...",
                key=f"article_{article.get('id') or idx}",
                use_container_width=True
            ):
                st.session_state.selected_article = article
            
            st.caption(f"{article.get('source', 'Unknown')} • {article.get('date', '')}")
            st.divider()
        
        # Display selected article details
        if st.session_state.selected_article:
            st.subheader("📄 Article Details")
            display_article(st.session_state.selected_article)
    elif not search_button and not st.session_state.search_results:
        st.info("👈 Use the search in the sidebar to find news articles")
    
    # Display environment status in the sidebar
    with st.sidebar.expander("Environment Status"):
        st.code(f"""
        Environment: {os.getenv('ENVIRONMENT', 'development')}
        MCP Server: {MCP_SERVER_URL}
        """)

if __name__ == "__main__":
    main()
