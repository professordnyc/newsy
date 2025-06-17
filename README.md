# Newsy

A headlines interrogation and classification agent.

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
   ```

4. **Set up environment variables**
   Copy `.env.example` to `.env` and fill in your API keys.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
newsy/
├── .gitignore
├── README.md
├── requirements.txt
├── .env.example
├── app.py              # Main Streamlit app
└── src/               # Source code
    ├── __init__.py
    ├── mcp/           # MCP server implementation
    └── services/      # Business logic
```

## License

MIT
