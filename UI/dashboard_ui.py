"""
Dashboard UI components for rendering the cryptocurrency analysis interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math

class Dashboard:
    """Main dashboard class for rendering analysis results."""
    
    def __init__(self):
        self.setup_styling()
    
    def setup_styling(self):
        """Setup custom CSS styling for the dashboard."""
        st.markdown("""
        <style>
        .metric-container {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .recommendation-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .buy-badge {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        
        .sell-badge {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        
        .hold-badge {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fed7aa;
        }
        
        .sentiment-bar {
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .risk-indicator {
            padding: 0.3rem 0.6rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .risk-low { background: #dcfce7; color: #166534; }
        .risk-medium { background: #fef3c7; color: #92400e; }
        .risk-high { background: #fee2e2; color: #991b1b; }
        </style>
        """, unsafe_allow_html=True)

def render_dashboard(analysis_result: Dict, user_query: str):
    """
    Render the main dashboard with analysis results.
    
    Args:
        analysis_result: Dictionary containing all analysis results
        user_query: Original user query
    """
    if not analysis_result.get("success"):
        st.error("Analysis failed. Please try again.")
        return
    
    dashboard = Dashboard()
    
    # Header section
    _render_header(analysis_result, user_query)
    
    # Main metrics overview
    _render_metrics_overview(analysis_result)
    
    st.divider()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary & Recommendation", 
        "📈 Price Analysis", 
        "📰 Sentiment & News",
        "🔮 Forecasts"
    ])
    
    with tab1:
        _render_summary_tab(analysis_result)
    
    with tab2:
        _render_price_analysis_tab(analysis_result)
    
    with tab3:
        _render_sentiment_tab(analysis_result)
    
    with tab4:
        _render_forecast_tab(analysis_result)
    
    # Risk disclaimer
    _render_risk_disclaimer()

def _render_header(analysis_result: Dict, user_query: str):
    """Render the dashboard header."""
    market_data = analysis_result.get("market", {})
    coin_name = market_data.get("coin", "Unknown").capitalize()
    symbol = market_data.get("symbol", "N/A")
    price = market_data.get("price_usd", 0)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(f"{coin_name} ({symbol}) Analysis")
        st.caption(f"Query: *{user_query}*")
        st.caption(f"Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.metric(
            label="Current Price",
            value=f"${price:,.2f}",
            delta=f"{market_data.get('pct_change_24h', 0):+.2f}%"
        )

def _render_metrics_overview(analysis_result: Dict):
    """Render key metrics overview."""
    market_data = analysis_result.get("market", {})
    sentiment_data = analysis_result.get("sentiment", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Market Cap",
            f"${market_data.get('market_cap', 0):,.0f}",
            help="Total market capitalization"
        )
    
    with col2:
        st.metric(
            "24h Volume", 
            f"${market_data.get('volume_24h', 0):,.0f}",
            help="24-hour trading volume"
        )
    
    with col3:
        sentiment_score = sentiment_data.get('aggregate_score', 0)
        sentiment_label = _get_sentiment_label(sentiment_score)
        st.metric(
            "Sentiment Score",
            f"{sentiment_score:+.3f}",
            delta=sentiment_label,
            help="News sentiment score (-1 to +1)"
        )
    
    with col4:
        rsi = market_data.get('rsi_14')
        rsi_display = f"{rsi:.1f}" if rsi and not math.isnan(rsi) else "N/A"
        st.metric(
            "RSI (14)",
            rsi_display,
            help="Relative Strength Index"
        )

def _render_summary_tab(analysis_result: Dict):
    """Render the summary and recommendation tab."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("AI Recommendation")
        
        ai_insights = analysis_result.get("ai_insights", {})
        recommendation = ai_insights.get("recommendation", "HOLD")
        confidence = ai_insights.get("confidence_score", 0.5)
        
        # Recommendation badge
        badge_class = _get_recommendation_badge_class(recommendation)
        st.markdown(
            f'<div class="recommendation-badge {badge_class}">{recommendation}</div>',
            unsafe_allow_html=True
        )
        
        # Confidence indicator
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
        
        # Detailed analysis
        analysis_text = ai_insights.get("detailed_analysis", "No detailed analysis available.")
        st.write(analysis_text)
        
        # Key factors
        key_factors = ai_insights.get("key_factors", [])
        if key_factors:
            st.subheader("Key Factors")
            for factor in key_factors[:3]:
                st.write(f"• {factor}")
    
    with col2:
        st.subheader("Risk Assessment")
        
        # Risk indicators
        risk_factors = ai_insights.get("risk_assessment", [])
        if risk_factors:
            for risk in risk_factors:
                risk_level = _assess_risk_level(risk)
                st.markdown(
                    f'<div class="risk-indicator risk-{risk_level}">{risk}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.write("No specific risks identified.")
        
        # Market conditions summary
        st.subheader("Market Summary")
        market_data = analysis_result.get("market", {})
        
        metrics = {
            "7d Change": f"{market_data.get('pct_change_7d', 0):+.2f}%",
            "Volatility": _calculate_volatility_label(market_data),
            "Trend": _get_trend_label(market_data)
        }
        
        for metric, value in metrics.items():
            st.write(f"**{metric}**: {value}")

def _render_price_analysis_tab(analysis_result: Dict):
    """Render price analysis tab with charts."""
    price_history = analysis_result.get("history")
    market_data = analysis_result.get("market", {})
    
    if price_history is not None and not price_history.empty:
        # Price chart
        st.subheader("Price Chart")
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=price_history.index,
            y=price_history['price'],
            mode='lines',
            name='Price',
            line=dict(color='#2E86C1', width=2)
        ))
        
        # Add moving averages if enough data
        if len(price_history) >= 7:
            ma7 = price_history['price'].rolling(7).mean()
            fig.add_trace(go.Scatter(
                x=price_history.index,
                y=ma7,
                mode='lines',
                name='MA7',
                line=dict(color='#F39C12', width=1, dash='dash')
            ))
        
        if len(price_history) >= 30:
            ma30 = price_history['price'].rolling(30).mean()
            fig.add_trace(go.Scatter(
                x=price_history.index,
                y=ma30,
                mode='lines',
                name='MA30',
                line=dict(color='#E74C3C', width=1, dash='dot')
            ))
        
        fig.update_layout(
            title="Price History with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Technical Indicators")
            
            rsi = market_data.get('rsi_14')
            if rsi and not math.isnan(rsi):
                st.write(f"**RSI (14)**: {rsi:.1f}")
                if rsi >= 70:
                    st.warning("⚠️ Potentially overbought")
                elif rsi <= 30:
                    st.success("✅ Potentially oversold")
                else:
                    st.info("ℹ️ Neutral territory")
        
        with col2:
            st.subheader("Price Levels")
            
            recent_high = price_history['price'].tail(30).max()
            recent_low = price_history['price'].tail(30).min()
            current_price = market_data.get('price_usd', 0)
            
            st.write(f"**30D High**: ${recent_high:.2f}")
            st.write(f"**30D Low**: ${recent_low:.2f}")
            st.write(f"**Current**: ${current_price:.2f}")
            
            # Position within range
            if recent_high != recent_low:
                position = (current_price - recent_low) / (recent_high - recent_low)
                st.progress(position, text=f"Position in range: {position:.1%}")
    
    else:
        st.warning("Price history data not available.")

def _render_sentiment_tab(analysis_result: Dict):
    """Render sentiment analysis tab."""
    sentiment_data = analysis_result.get("sentiment", {})
    articles = analysis_result.get("articles", [])
    
    st.subheader("News Sentiment Analysis")
    
    # Sentiment overview
    col1, col2 = st.columns([1, 2])
    
    with col1:
        percentages = sentiment_data.get("percentages", {})
        
        st.write("**Sentiment Distribution:**")
        st.write(f"Positive: {percentages.get('positive', 0):.1f}%")
        st.write(f"Neutral: {percentages.get('neutral', 0):.1f}%")  
        st.write(f"Negative: {percentages.get('negative', 0):.1f}%")
        
        # Overall sentiment
        overall_score = sentiment_data.get('aggregate_score', 0)
        sentiment_label = _get_sentiment_label(overall_score)
        st.metric("Overall Sentiment", f"{overall_score:+.3f}", sentiment_label)
    
    with col2:
        # Sentiment pie chart
        if percentages:
            fig = px.pie(
                values=list(percentages.values()),
                names=list(percentages.keys()),
                title="Sentiment Distribution",
                color_discrete_map={
                    'positive': '#00D4AA',
                    'neutral': '#FFC107', 
                    'negative': '#FF6B6B'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent headlines
    st.subheader("Recent Headlines")
    
    sentiment_df = sentiment_data.get("sentiment_dataframe")
    if sentiment_df is not None and not sentiment_df.empty:
        # Display top headlines with sentiment
        display_df = sentiment_df.head(10).copy()
        display_df['sentiment'] = display_df.apply(
            lambda x: f"{x['label'].capitalize()} ({x['confidence']:.2f})", axis=1
        )
        
        st.dataframe(
            display_df[['text', 'sentiment']],
            column_config={
                'text': 'Headline',
                'sentiment': 'Sentiment (Confidence)'
            },
            use_container_width=True
        )
    else:
        st.info("No recent headlines available for analysis.")

def _render_forecast_tab(analysis_result: Dict):
    """Render forecasting tab."""
    forecast_data = analysis_result.get("forecasts", {})
    market_data = analysis_result.get("market", {})
    
    st.subheader("Price Forecasts")
    
    # Check available forecasts
    has_prophet = forecast_data.get("prophet_forecast") is not None
    has_lstm = forecast_data.get("lstm_forecast") is not None
    has_ensemble = forecast_data.get("ensemble_forecast") is not None
    
    if not (has_prophet or has_lstm or has_ensemble):
        st.warning("No forecasts available. This may be due to insufficient historical data.")
        return
    
    # Forecast comparison
    forecast_dates = forecast_data.get("forecast_dates", [])
    current_price = market_data.get("price_usd", 0)
    
    if forecast_dates:
        # Create forecast comparison chart
        fig = go.Figure()
        
        # Add current price as starting point
        if current_price > 0:
            fig.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[current_price],
                mode='markers',
                name='Current Price',
                marker=dict(color='black', size=10)
            ))
        
        # Add forecasts
        if has_prophet:
            prophet_values = forecast_data["prophet_forecast"]["values"]
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=prophet_values,
                mode='lines+markers',
                name='Prophet',
                line=dict(color='#3498DB')
            ))
        
        if has_lstm:
            lstm_values = forecast_data["lstm_forecast"]["values"]
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=lstm_values,
                mode='lines+markers',
                name='LSTM',
                line=dict(color='#E74C3C')
            ))
        
        if has_ensemble:
            ensemble_values = forecast_data["ensemble_forecast"]["values"]
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=ensemble_values,
                mode='lines+markers',
                name='Ensemble',
                line=dict(color='#2ECC71', width=3)
            ))
        
        fig.update_layout(
            title="Price Forecasts Comparison",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary table
        st.subheader("Forecast Summary")
        
        summary_data = []
        
        if has_prophet and forecast_data["prophet_forecast"]["values"]:
            prophet_last = forecast_data["prophet_forecast"]["values"][-1]
            prophet_change = ((prophet_last - current_price) / current_price * 100) if current_price > 0 else 0
            summary_data.append({
                "Model": "Prophet",
                "7-Day Price": f"${prophet_last:.2f}",
                "Expected Change": f"{prophet_change:+.2f}%"
            })
        
        if has_lstm and forecast_data["lstm_forecast"]["values"]:
            lstm_last = forecast_data["lstm_forecast"]["values"][-1]
            lstm_change = ((lstm_last - current_price) / current_price * 100) if current_price > 0 else 0
            summary_data.append({
                "Model": "LSTM",
                "7-Day Price": f"${lstm_last:.2f}",
                "Expected Change": f"{lstm_change:+.2f}%"
            })
        
        if has_ensemble and forecast_data["ensemble_forecast"]["values"]:
            ensemble_last = forecast_data["ensemble_forecast"]["values"][-1]
            ensemble_change = ((ensemble_last - current_price) / current_price * 100) if current_price > 0 else 0
            summary_data.append({
                "Model": "Ensemble",
                "7-Day Price": f"${ensemble_last:.2f}",
                "Expected Change": f"{ensemble_change:+.2f}%"
            })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

def _render_risk_disclaimer():
    """Render risk disclaimer section."""
    with st.expander("⚠️ Important Risk Disclaimer"):
        st.warning("""
        **Educational Purpose Only**: This analysis is for educational and research purposes only. 
        It should not be considered as financial advice.
        
        **High Risk Investment**: Cryptocurrency investments carry significant risks including:
        - Extreme price volatility
        - Regulatory uncertainty
        - Market manipulation risks
        - Technical and security risks
        
        **No Guarantees**: Past performance does not guarantee future results. AI predictions 
        and forecasts are based on historical data and may not accurately predict future prices.
        
        **Professional Advice**: Always consult with qualified financial advisors before making 
        investment decisions. Only invest what you can afford to lose.
        """)

# Helper functions
def _get_sentiment_label(score: float) -> str:
    """Get sentiment label from score."""
    if score > 0.2:
        return "Very Positive"
    elif score > 0.05:
        return "Positive"
    elif score > -0.05:
        return "Neutral"
    elif score > -0.2:
        return "Negative"
    else:
        return "Very Negative"

def _get_recommendation_badge_class(recommendation: str) -> str:
    """Get CSS class for recommendation badge."""
    if "BUY" in recommendation.upper():
        return "buy-badge"
    elif "SELL" in recommendation.upper():
        return "sell-badge"
    else:
        return "hold-badge"

def _assess_risk_level(risk_text: str) -> str:
    """Assess risk level from risk text."""
    risk_lower = risk_text.lower()
    if any(word in risk_lower for word in ["high", "significant", "major", "severe"]):
        return "high"
    elif any(word in risk_lower for word in ["moderate", "medium", "some"]):
        return "medium"
    else:
        return "low"

def _calculate_volatility_label(market_data: Dict) -> str:
    """Calculate volatility label from market data."""
    change_24h = abs(market_data.get('pct_change_24h', 0))
    
    if change_24h > 10:
        return "Very High"
    elif change_24h > 5:
        return "High"
    elif change_24h > 2:
        return "Moderate"
    else:
        return "Low"

def _get_trend_label(market_data: Dict) -> str:
    """Get trend label from market data."""
    change_7d = market_data.get('pct_change_7d', 0)
    
    if change_7d > 10:
        return "Strong Uptrend"
    elif change_7d > 2:
        return "Uptrend"
    elif change_7d > -2:
        return "Sideways"
    elif change_7d > -10:
        return "Downtrend"
    else:
        return "Strong Downtrend"