# Newsy

A headlines interrogation and classification agent that uses Clarifai's Llama-3 model to identify leading vs. under-reported news stories.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/professordnyc/newsy.git
   cd newsy
   ```

2. **Set up virtual environment**
   - Windows:
     ```bash
     python -m venv venv
   - macOS/Linux:
     ```bash
     python3 -m venv venv
   ```
   
   Activate the environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   ```
   Then edit `.env` and add your API keys:
   - `CLARIFAI_API_KEY`
   - `SERPAPI_API_KEY`

5. **Run the application**
   Start the backend server:
   ```bash
   python run_server.py
   ```
   
   In a new terminal, start the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py` - Streamlit frontend application
- `run_server.py` - Backend server launcher
- `mcp/` - Backend API and services
  - `v1/` - API version 1
    - `src/` - Source code
      - `main.py` - FastAPI application and endpoints
      - `services/` - Service implementations
- `requirements.txt` - Python dependencies
- `.env.example` - Example environment variables

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

## Contributing

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your commit message"
   ```

3. Push your changes and create a pull request

## Project Structure (Detailed)

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
