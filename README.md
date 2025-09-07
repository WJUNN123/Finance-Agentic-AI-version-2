# Crypto Analysis Dashboard

A comprehensive cryptocurrency analysis tool that combines real-time market data, sentiment analysis, and AI-powered insights to help users make informed decisions about cryptocurrency investments.

## Features

- **Real-time Market Data**: Live prices, market caps, and trading volumes from CoinGecko API
- **Sentiment Analysis**: AI-powered analysis of crypto news using FinBERT model
- **Price Forecasting**: Multiple forecasting models including Prophet and LSTM
- **AI Insights**: Personalized investment recommendations using Google Gemini
- **Interactive Dashboard**: Clean, responsive Streamlit interface
- **Risk Analysis**: Comprehensive risk assessment and disclaimers

## Screenshots

![Dashboard Overview](docs/dashboard-screenshot.png)
*Main dashboard showing Bitcoin analysis with price charts, sentiment data, and AI recommendations*

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-analysis-dashboard.git
cd crypto-analysis-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory or use Streamlit secrets:

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional, for private models
```

Or create `.streamlit/secrets.toml`:

```toml
[gemini]
api_key = "your_gemini_api_key_here"

HF_TOKEN = "your_huggingface_token_here"
```

### 3. Run the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Examples

### Basic Queries
- "Bitcoin price forecast"
- "Should I buy Ethereum?"
- "Solana market sentiment"
- "Cardano 7-day prediction"

### Advanced Analysis
- "Compare Bitcoin and Ethereum performance"
- "What are the risks of investing in Solana?"
- "Bitcoin technical analysis with RSI"

## Project Structure

```
crypto-analysis-dashboard/
│
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env.example           # Environment variables template
│
├── core/                  # Core business logic
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_fetcher.py    # Market data and news fetching
│   ├── sentiment_analyzer.py  # FinBERT sentiment analysis
│   ├── forecaster.py      # Prophet and LSTM forecasting
│   ├── ai_advisor.py      # Google Gemini integration
│   └── memory_manager.py  # Session and long-term memory
│
├── ui/                    # User interface components
│   ├── __init__.py
│   └── dashboard.py       # Dashboard rendering functions
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── helpers.py         # Helper functions and parsers
│
├── models/                # Saved ML models (created at runtime)
├── data/                  # Data storage (created at runtime)
│   ├── memory.db          # SQLite database for conversations
│   └── cache/             # Temporary cache files
│
└── docs/                  # Documentation and screenshots
    └── screenshots/
```

## Configuration Options

### Supported Cryptocurrencies

The app supports analysis for major cryptocurrencies including:
- Bitcoin (BTC)
- Ethereum (ETH) 
- Solana (SOL)
- BNB (BNB)
- XRP (XRP)
- Cardano (ADA)
- Dogecoin (DOGE)

### Forecasting Models

1. **Prophet**: Facebook's time series forecasting tool
   - Good for trend detection and seasonality
   - Handles missing data well
   - Provides uncertainty intervals

2. **LSTM**: Long Short-Term Memory neural networks
   - Deep learning approach for price prediction
   - Captures complex patterns in price data
   - Uses technical indicators as features

3. **Ensemble**: Combines Prophet and LSTM predictions
   - Weighted average of both models
   - Generally more robust than individual models

## API Keys Setup

### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Create a new API key
3. Add to your `.env` file or Streamlit secrets

### Hugging Face Token (Optional)
1. Create account at [Hugging Face](https://huggingface.co/)
2. Go to Settings > Access Tokens
3. Create a new token with read permissions

## Development

### Adding New Cryptocurrencies

Edit `core/config.py` and add to the `SUPPORTED_COINS` list:

```python
{"name": "Polygon", "id": "matic-network", "symbol": "matic"}
```

### Customizing AI Responses

Modify the prompt in `core/ai_advisor.py` to change how the AI generates insights:

```python
def generate_insights(self, ...):
    prompt = f"""
    Your custom prompt here...
    Market data: {market_data}
    """
```

### Adding New Data Sources

Extend `core/data_fetcher.py` to add new RSS feeds or data providers:

```python
RSS_FEEDS = [
    "https://your-new-feed.com/rss",
    # ... existing feeds
]
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your API keys in the Streamlit Cloud secrets manager
4. Deploy!

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku Deployment

1. Create `Procfile`:
```
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write tests for new functionality
- Update documentation for new features

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_data_fetcher.py -v
```

## Performance Optimization

### Caching

The app uses Streamlit's caching features:
- Market data is cached for 5 minutes
- News articles are cached for 15 minutes
- Model predictions are cached for 1 hour

### Memory Management

- LSTM models are saved to disk after training
- Long-term memory uses FAISS for efficient similarity search
- Session data is stored in SQLite database

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **API Rate Limiting**: CoinGecko free API has rate limits
   - Wait between requests
   - Consider upgrading to pro API for higher limits

3. **Memory Issues with LSTM**: Reduce batch size or window size in config
   ```python
   FORECAST_PARAMS = {
       "lstm_window": 20,  # Reduced from 30
       "lstm_batch_size": 8  # Reduced from 16
   }
   ```

4. **Slow Sentiment Analysis**: FinBERT model loads slowly on first use
   - Models are cached after first load
   - Consider using GPU for faster inference

### Getting Help

- Check the [Issues](https://github.com/yourusername/crypto-analysis-dashboard/issues) page
- Review the [Discussions](https://github.com/yourusername/crypto-analysis-dashboard/discussions) for Q&A
- Read the [Wiki](https://github.com/yourusername/crypto-analysis-dashboard/wiki) for detailed guides

## Disclaimer

**Important**: This application is for educational and research purposes only. It is not intended as financial advice. Cryptocurrency investments carry significant risks including:

- **High Volatility**: Crypto prices can fluctuate dramatically
- **Regulatory Risk**: Changing regulations may affect crypto markets
- **Technical Risk**: Smart contracts and exchanges may have vulnerabilities
- **Market Risk**: Crypto markets are largely unregulated and speculative

**Always do your own research** and consult with qualified financial advisors before making investment decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CoinGecko](https://www.coingecko.com/) for cryptocurrency market data
- [Hugging Face](https://huggingface.co/) for the FinBERT sentiment analysis model
- [Google](https://ai.google.dev/) for the Gemini AI API
- [Streamlit](https://streamlit.io/) for the web framework
- [Prophet](https://facebook.github.io/prophet/) by Facebook for time series forecasting

## Changelog

### v1.0.0 (2024-01-XX)
- Initial release
- Basic cryptocurrency analysis
- Sentiment analysis integration
- AI-powered insights
- Streamlit dashboard

### v1.1.0 (Planned)
- Portfolio tracking features
- Historical backtesting
- Additional technical indicators
- Multi-language support

---

Made with ❤️ by [Your Name](https://github.com/yourusername)

⭐ Star this repo if you find it helpful!