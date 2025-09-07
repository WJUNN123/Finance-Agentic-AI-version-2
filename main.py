"""
Crypto Analysis Streamlit App
============================
A comprehensive cryptocurrency analysis tool with sentiment analysis, 
price forecasting, and AI-powered insights.

Features:
- Real-time market data from CoinGecko
- News sentiment analysis using FinBERT
- Price forecasting with Prophet and LSTM
- AI-powered insights using Google Gemini
- Interactive dashboard with Streamlit
"""

import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, List

# Import our modular components
from core.Config import AppConfig, UI_CONFIG
from core.Data_fetcher import DataFetcher
from core.Sentiment_analyzer import SentimentAnalyzer
from core.forecaster import Forecaster
from core.ai_advisor import AIAdvisor
from core.memory_manager import MemoryManager
from ui.dashboard import render_dashboard
from utils.helpers import parse_user_message, format_response

class CryptoAnalysisApp:
    """Main application class that orchestrates all components."""
    
    def __init__(self):
        self.config = AppConfig()
        self.data_fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.forecaster = Forecaster()
        self.ai_advisor = AIAdvisor()
        self.memory_manager = MemoryManager()
        
    def analyze_cryptocurrency(self, coin_id: str, coin_symbol: str, 
                             horizon_days: int = 7) -> Dict:
        """
        Main analysis pipeline that combines all components.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            coin_symbol: Coin symbol (e.g., 'BTC')
            horizon_days: Forecast horizon in days
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Fetch market data
            market_data = self.data_fetcher.get_market_data([coin_id])
            if market_data.empty:
                return {"error": f"No market data found for {coin_id}"}
                
            price_history = self.data_fetcher.get_price_history(coin_id, days=180)
            
            # Get news and sentiment
            articles = self.data_fetcher.get_news_articles(coin_symbol, coin_id)
            sentiment_results = self.sentiment_analyzer.analyze_articles(articles)
            
            # Generate forecasts
            forecasts = self.forecaster.generate_forecasts(
                price_history, horizon_days
            )
            
            # Get AI insights
            ai_insights = self.ai_advisor.generate_insights(
                coin_id=coin_id,
                coin_symbol=coin_symbol,
                market_data=market_data.iloc[0],
                sentiment_data=sentiment_results,
                forecast_data=forecasts,
                horizon_days=horizon_days
            )
            
            # Combine all results
            return {
                "market": self._format_market_data(market_data.iloc[0]),
                "history": price_history,
                "articles": articles,
                "sentiment": sentiment_results,
                "forecasts": forecasts,
                "ai_insights": ai_insights,
                "success": True
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _format_market_data(self, row) -> Dict:
        """Format market data for consistent output."""
        return {
            "coin": row.get("id", ""),
            "symbol": row.get("symbol", "").upper(),
            "price_usd": float(row.get("current_price", 0)),
            "pct_change_24h": row.get("price_change_percentage_24h"),
            "pct_change_7d": row.get("price_change_percentage_7d_in_currency"),
            "market_cap": float(row.get("market_cap", 0)),
            "volume_24h": float(row.get("total_volume", 0)),
            "rsi_14": self.data_fetcher.calculate_rsi(row) if hasattr(self.data_fetcher, 'calculate_rsi') else None
        }

def setup_streamlit_ui():
    """Configure Streamlit page settings and custom CSS."""
    st.set_page_config(
        page_title="Crypto Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Apply custom CSS
    st.markdown(UI_CONFIG["custom_css"], unsafe_allow_html=True)
    
    # Header
    st.markdown(UI_CONFIG["header_html"], unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

def main():
    """Main application entry point."""
    # Setup UI
    setup_streamlit_ui()
    initialize_session_state()
    
    # Initialize app
    app = CryptoAnalysisApp()
    
    # User input section
    st.markdown("### Ask about any cryptocurrency")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "",
            placeholder="E.g., 'Bitcoin 7-day forecast' or 'Should I buy Ethereum?'",
            key="crypto_query"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze", use_container_width=True)
    
    # Quick action buttons
    st.markdown("**Quick Actions:**")
    quick_buttons = st.columns(4)
    quick_queries = [
        "Bitcoin forecast",
        "Ethereum sentiment", 
        "Should I buy Solana?",
        "Cardano price analysis"
    ]
    
    for i, query in enumerate(quick_queries):
        if quick_buttons[i].button(query, key=f"quick_{i}"):
            st.session_state.crypto_query = query
            user_input = query
            analyze_button = True
    
    # Process analysis request
    if analyze_button and user_input.strip():
        with st.spinner("Analyzing cryptocurrency data..."):
            # Parse user input
            parsed_input = parse_user_message(user_input)
            
            # Run analysis
            result = app.analyze_cryptocurrency(
                coin_id=parsed_input["coin_id"],
                coin_symbol=parsed_input["coin_symbol"], 
                horizon_days=parsed_input["horizon_days"]
            )
            
            # Store results
            st.session_state.analysis_result = result
            st.session_state.last_query = user_input
            
            # Save to memory
            app.memory_manager.save_interaction(
                session_id=st.session_state.session_id,
                query=user_input,
                result=result
            )
    
    # Display results
    if st.session_state.analysis_result:
        if st.session_state.analysis_result.get("success"):
            render_dashboard(
                st.session_state.analysis_result,
                st.session_state.last_query
            )
        else:
            st.error(f"Analysis failed: {st.session_state.analysis_result.get('error')}")
    else:
        # Show welcome message
        st.info(
            "👋 Welcome! Enter a cryptocurrency name or question above to get started. "
            "This app provides market analysis, sentiment insights, and price forecasts."
        )
        
        # Show example queries
        with st.expander("📝 Example Queries"):
            st.markdown("""
            - **Price Analysis**: "Bitcoin current price", "Ethereum market cap"
            - **Forecasts**: "Bitcoin 7-day forecast", "SOL price prediction"  
            - **Investment Advice**: "Should I buy Bitcoin?", "Is Ethereum a good investment?"
            - **Sentiment**: "Bitcoin news sentiment", "Ethereum market sentiment"
            - **Comparisons**: "Bitcoin vs Ethereum", "Best crypto to buy now"
            """)

if __name__ == "__main__":
    main()
