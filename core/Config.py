"""
Configuration settings for the Crypto Analysis App
"""

import os
import streamlit as st

class AppConfig:
    """Central configuration class for the application."""
    
    # Supported cryptocurrencies
    SUPPORTED_COINS = [
        {"name": "Bitcoin", "id": "bitcoin", "symbol": "btc"},
        {"name": "Ethereum", "id": "ethereum", "symbol": "eth"},
        {"name": "Solana", "id": "solana", "symbol": "sol"},
        {"name": "BNB", "id": "binancecoin", "symbol": "bnb"},
        {"name": "XRP", "id": "ripple", "symbol": "xrp"},
        {"name": "Cardano", "id": "cardano", "symbol": "ada"},
        {"name": "Dogecoin", "id": "dogecoin", "symbol": "doge"},
    ]
    
    # News RSS feeds
    RSS_FEEDS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.google.com/rss/search?q=cryptocurrency&hl=en-US&gl=US&ceid=US:en",
    ]
    
    # AI Model settings
    FINBERT_MODEL = "ProsusAI/finbert"
    SENTIMENT_DEVICE = -1  # CPU: -1, GPU: 0
    
    # Forecasting parameters
    FORECAST_PARAMS = {
        "lstm_window": 30,
        "lstm_epochs": 20,
        "lstm_batch_size": 16,
        "prophet_seasonality": "multiplicative"
    }
    
    @property
    def coin_name_mapping(self):
        """Mapping of coin names to CoinGecko IDs."""
        return {coin["name"].lower(): coin["id"] for coin in self.SUPPORTED_COINS}
    
    @property
    def coin_symbol_mapping(self):
        """Mapping of coin symbols to CoinGecko IDs."""
        return {coin["symbol"].lower(): coin["id"] for coin in self.SUPPORTED_COINS}
    
    @staticmethod
    def get_api_key(service: str) -> str:
        """Get API key from Streamlit secrets or environment variables."""
        try:
            # Try Streamlit secrets first
            return st.secrets.get(service, {}).get("api_key", "")
        except:
            # Fall back to environment variables
            return os.getenv(f"{service.upper()}_API_KEY", "")

# UI Configuration
UI_CONFIG = {
    "custom_css": """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .app-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Recommendation badges */
    .rec-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin: 8px 0;
    }
    
    .rec-buy {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .rec-sell {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .rec-hold {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fed7aa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
    }
    
    /* Quick action buttons */
    .quick-action {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .quick-action:hover {
        background: #e2e8f0;
    }
    </style>
    """,
    
    "header_html": """
    <div class='app-header'>
        <div class='app-logo'>📊</div>
        <div>
            <h1 style='margin: 0; color: #1e293b;'>Crypto Analysis Dashboard</h1>
            <p style='margin: 0; color: #64748b; font-size: 1.1rem;'>
                AI-powered cryptocurrency analysis and forecasting
            </p>
        </div>
    </div>
    """
}

# Risk disclaimers and warnings
RISK_DISCLAIMERS = {
    "main": """
    ⚠️ **Important Disclaimer**: This tool is for educational purposes only. 
    Cryptocurrency investments carry significant risks. Always do your own research 
    and consider consulting with a financial advisor before making investment decisions.
    """,
    
    "ai_limitation": """
    🤖 **AI Limitations**: AI-generated insights are based on historical data and 
    current market sentiment. They cannot predict future market movements with certainty.
    """,
    
    "data_accuracy": """
    📊 **Data Accuracy**: While we strive for accuracy, market data may have delays 
    or inaccuracies. Always verify information from multiple sources.
    """
}