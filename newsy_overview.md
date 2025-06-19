# Newsy Architecture Overview

## System Architecture

Newsy is a news classification system with a client-server architecture:

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│                 │     │                   │     │                  │
│  Streamlit UI   │────▶│    FastAPI MCP     │◀───▶│  External APIs   │
│   (Frontend)    │     │    (Backend)      │     │ (SerpAPI, Clarifai)
│                 │     │                   │     │                  │
└─────────────────┘     └───────────────────┘     └──────────────────┘
```

## 1. Backend (MCP - Model Context Protocol)

The backend is built with FastAPI and provides several key services:

### Core Services

#### `SerpAPIService` (`src/services/serpapi_service.py`)
```python
class SerpAPIService:
    def search_news(self, query: str, region: str = "us", max_results: int = 10) -> List[Dict]:
        """
        Search for news articles using SerpAPI.
        
        Args:
            query: Search query string
            region: Region code (default: 'us')
            max_results: Maximum number of results to return
            
        Returns:
            List of news article dictionaries
        """
        # Implementation...
```

#### `ClarifaiService` (`src/services/clarifai_service.py`)
```python
class ClarifaiService:
    def analyze_article(self, article: Dict, classification_type: ClassificationType) -> Dict:
        """
        Analyze article content using Clarifai's AI models.
        
        Args:
            article: Dictionary containing article data
            classification_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        # Implementation...
```

### API Endpoints (`mcp/v1/src/main.py`)

Key endpoints:

1. **Search & Classification**
   - `POST /search/headlines` - Basic news search
   - `POST /search/headlines/classified` - Search with AI classification
   - `POST /articles/analyze` - Analyze article content

2. **System**
   - `GET /health` - Service health check
   - `POST /cache/clear` - Clear the cache
   - `GET /test/serpapi` - Test SerpAPI integration

Example endpoint implementation:
```python
@app.post("/search/headlines/classified")
async def search_headlines_classified(
    request: SearchRequest,
    services: dict = Depends(check_services)
):
    """
    Search for news headlines and classify them in a single operation.
    
    Returns:
        List of news articles with classifications
    """
    # Implementation...
```

## 2. Frontend (Streamlit)

The frontend is built with Streamlit and provides a simple UI for interacting with the backend.

### Key Components

1. **Search Interface** (`app.py`)
   - Search box with query input
   - Region selection
   - Result count selector

2. **Results Display**
   - Classified articles (Leading/Under-reported)
   - Article metadata (source, date, etc.)
   - Links to full articles

## 3. Data Flow

1. User enters search query in Streamlit UI
2. Frontend sends request to `/search/headlines/classified`
3. Backend:
   - Fetches news from SerpAPI
   - Processes and classifies articles using Clarifai
   - Caches results
   - Returns classified articles
4. Frontend displays results with visual indicators

## 4. Caching Strategy

```python
class CacheService:
    def __init__(self, cache_name: str = 'newsy_cache', expire_after: int = 86400):
        """
        Initialize the cache service.
        
        Args:
            cache_name: Name for the cache database
            expire_after: Cache expiration in seconds (default: 24h)
        """
        # Implementation...
```

## 5. Environment Configuration

Required environment variables (`.env`):
```
SERPAPI_KEY=your_serpapi_key
CLARIFAI_API_KEY=your_clarifai_key
```

## 6. Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the backend:
   ```bash
   python run_server.py
   ```

4. Run the frontend:
   ```bash
   streamlit run app.py
   ```

## 7. Testing

Run tests with:
```bash
pytest
```

## 8. Deployment

### Backend (Render)
1. Set up a new Web Service on Render
2. Use the `Procfile` for configuration
3. Set environment variables in the dashboard

### Frontend (Streamlit Cloud)
1. Connect your GitHub repository
2. Set `API_URL` to point to your backend
3. Deploy

## 9. Future Improvements

1. **Enhanced UI/UX**
   - Better visual design
   - More interactive elements
   - Improved mobile responsiveness

2. **Advanced Analytics**
   - Sentiment analysis
   - Topic modeling
   - Trend analysis

3. **Scalability**
   - Distributed caching
   - Load balancing
   - Rate limiting
