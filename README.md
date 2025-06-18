# Newsy

A headlines interrogation and classification agent that uses Clarifai's Llama-3 model to identify leading vs. under-reported news stories.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd newsy
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # For HTML cleaning functionality
   pip install "lxml[html_clean]"
   # For SerpAPI integration
   pip install serpapi
   ```

4. **Set up environment variables**
   Copy `.env.example` to `.env` and fill in your API keys.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Performance Optimizations

The application includes several performance optimizations to improve responsiveness and reduce API costs:

1. **In-memory Caching System**
   - Configurable cache size and TTL (Time-To-Live)
   - MD5 hashing for efficient cache keys
   - Automatic cache cleanup
   - Cache statistics tracking

2. **Parallel Processing**
   - Uses ThreadPoolExecutor for concurrent article classification
   - Configurable number of worker threads via UI
   - Significant speedup for batch operations

3. **Improved Error Handling**
   - Specific gRPC error detection
   - Configurable timeouts via UI
   - Graceful recovery from API failures

These optimizations are available in both the Streamlit UI and MCP API endpoints.

## Project Structure

```
newsy/
├── .gitignore
├── README.md
├── requirements.txt
├── .env.example
├── app.py              # Main Streamlit app
├── mcp/                # MCP server
│   └── v1/             # API version 1
│       └── src/        # Source code for MCP
└── src/                # Source code
    ├── __init__.py
    └── services/       # Business logic
        ├── cache_service.py
        ├── clarifai_service.py
        └── serpapi_service.py
```

## Troubleshooting

### Common Issues

1. **503 Service Unavailable Error**
   - Ensure all dependencies are installed, including `lxml[html_clean]` and `serpapi`
   - Activate the virtual environment before running the server
   - Check `newsy_mcp.log` for specific service initialization errors

2. **Connection Refused Error**
   - Make sure the server is running on port 8000
   - Check if there are any errors in the server logs
   - Restart the server if necessary

3. **500 Internal Server Error**
   - Check the server logs for specific error messages
   - Ensure all API keys are correctly set in the `.env` file
   - Verify that all services are initialized correctly via the `/health` endpoint

### Running the Server

To run the MCP server:
```bash
# Activate the virtual environment first
.\venv\Scripts\activate  # On Windows

# Run the server
python run_server.py
```

## License

MIT
