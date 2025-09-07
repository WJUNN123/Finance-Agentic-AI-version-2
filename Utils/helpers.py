"""
Utility helper functions for the crypto analysis application
"""

import re
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from core.config import AppConfig

def parse_user_message(message: str) -> Dict:
    """
    Parse user message to extract intent, coin, and parameters.
    
    Args:
        message: User input message
        
    Returns:
        Dictionary with parsed components
    """
    config = AppConfig()
    message_lower = message.lower().strip()
    
    # Extract coin information
    coin_id = extract_coin_from_message(message_lower, config)
    coin_symbol = get_coin_symbol(coin_id, config)
    
    # Extract intent
    intent = extract_intent(message_lower)
    
    # Extract time horizon
    horizon_days = extract_time_horizon(message_lower)
    
    # Extract risk preference
    risk_preference = extract_risk_preference(message_lower)
    
    return {
        "coin_id": coin_id,
        "coin_symbol": coin_symbol,
        "intent": intent,
        "horizon_days": horizon_days,
        "risk_preference": risk_preference,
        "original_message": message
    }

def extract_coin_from_message(message: str, config: AppConfig) -> str:
    """Extract cryptocurrency from user message."""
    message = message.lower()
    
    # Check for coin names
    for coin_name, coin_id in config.coin_name_mapping.items():
        if coin_name in message:
            return coin_id
    
    # Check for coin symbols
    for coin_symbol, coin_id in config.coin_symbol_mapping.items():
        # Use word boundaries to avoid partial matches
        if re.search(r'\b' + re.escape(coin_symbol) + r'\b', message):
            return coin_id
    
    # Default to Bitcoin if no specific coin mentioned
    return "bitcoin"

def get_coin_symbol(coin_id: str, config: AppConfig) -> str:
    """Get coin symbol from coin ID."""
    for coin in config.SUPPORTED_COINS:
        if coin["id"] == coin_id:
            return coin["symbol"].upper()
    return coin_id[:3].upper()  # Fallback

def extract_intent(message: str) -> str:
    """Extract user intent from message."""
    message = message.lower()
    
    # Investment advice intent
    if any(phrase in message for phrase in [
        "should i buy", "should i sell", "is it a good investment",
        "recommend", "advice", "what do you think"
    ]):
        return "investment_advice"
    
    # Price prediction intent
    elif any(phrase in message for phrase in [
        "forecast", "predict", "price prediction", "next week",
        "next month", "future price", "will it go up"
    ]):
        return "price_forecast"
    
    # Sentiment analysis intent
    elif any(phrase in message for phrase in [
        "sentiment", "news", "headlines", "market feeling",
        "what people think"
    ]):
        return "sentiment_analysis"
    
    # Current price intent
    elif any(phrase in message for phrase in [
        "current price", "price now", "how much", "current value"
    ]):
        return "current_price"
    
    # Technical analysis intent
    elif any(phrase in message for phrase in [
        "technical analysis", "rsi", "moving average", "support",
        "resistance", "chart analysis"
    ]):
        return "technical_analysis"
    
    # General analysis (default)
    else:
        return "general_analysis"

def extract_time_horizon(message: str) -> int:
    """Extract time horizon in days from message."""
    message = message.lower()
    
    # Look for specific day mentions
    day_match = re.search(r'(\d+)\s*days?', message)
    if day_match:
        return int(day_match.group(1))
    
    # Look for week mentions
    week_match = re.search(r'(\d+)\s*weeks?', message)
    if week_match:
        return int(week_match.group(1)) * 7
    
    # Look for month mentions
    month_match = re.search(r'(\d+)\s*months?', message)
    if month_match:
        return int(month_match.group(1)) * 30
    
    # Common time phrases
    if any(phrase in message for phrase in ["next week", "1 week", "one week"]):
        return 7
    elif any(phrase in message for phrase in ["next month", "1 month", "one month"]):
        return 30
    elif any(phrase in message for phrase in ["short term", "few days"]):
        return 3
    elif any(phrase in message for phrase in ["long term", "long run"]):
        return 90
    
    # Default to 7 days
    return 7

def extract_risk_preference(message: str) -> str:
    """Extract risk preference from message."""
    message = message.lower()
    
    if any(phrase in message for phrase in [
        "conservative", "low risk", "safe", "careful", "cautious"
    ]):
        return "low"
    elif any(phrase in message for phrase in [
        "aggressive", "high risk", "risky", "bold"
    ]):
        return "high"
    else:
        return "medium"

def generate_query_hash(coin_id: str, intent: str, horizon_days: int) -> str:
    """Generate hash for caching query results."""
    query_string = f"{coin_id}_{intent}_{horizon_days}"
    return hashlib.md5(query_string.encode()).hexdigest()

def format_currency(amount: Union[int, float], decimals: int = 2) -> str:
    """Format currency amounts with appropriate suffixes."""
    if pd.isna(amount) or amount == 0:
        return "N/A"
    
    abs_amount = abs(amount)
    
    if abs_amount >= 1_000_000_000_000:
        return f"${amount/1_000_000_000_000:.{decimals}f}T"
    elif abs_amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.{decimals}f}B"
    elif abs_amount >= 1_000_000:
        return f"${amount/1_000_000:.{decimals}f}M"
    elif abs_amount >= 1_000:
        return f"${amount/1_000:.{decimals}f}K"
    else:
        return f"${amount:.{decimals}f}"

def format_percentage(value: Union[int, float], decimals: int = 2) -> str:
    """Format percentage values."""
    if pd.isna(value):
        return "N/A"
    return f"{value:+.{decimals}f}%"

def format_timestamp(timestamp: Union[int, float, str]) -> str:
    """Format timestamp to human-readable string."""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except:
        return "Unknown"

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform safe division with default value for zero division."""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default

def calculate_price_change_percentage(current_price: float, previous_price: float) -> float:
    """Calculate percentage change between two prices."""
    if previous_price == 0 or pd.isna(previous_price) or pd.isna(current_price):
        return 0.0
    return ((current_price - previous_price) / previous_price) * 100

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame contains required columns."""
    if df is None or df.empty:
        return False
    return all(col in df.columns for col in required_columns)

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove problematic characters
    text = re.sub(r'[^\w\s\-.,!?()%$]', '', text)
    
    return text

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def get_trend_direction(values: List[float]) -> str:
    """Determine trend direction from a list of values."""
    if not values or len(values) < 2:
        return "unknown"
    
    # Calculate simple linear trend
    x = list(range(len(values)))
    slope = np.polyfit(x, values, 1)[0]
    
    if slope > 0.01:
        return "upward"
    elif slope < -0.01:
        return "downward"
    else:
        return "sideways"

def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """Calculate price volatility using rolling standard deviation."""
    if len(prices) < window:
        return 0.0
    
    price_series = pd.Series(prices)
    returns = price_series.pct_change().dropna()
    
    if len(returns) < window:
        return 0.0
    
    return float(returns.rolling(window).std().iloc[-1])

def normalize_score(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Normalize a score to be between min_val and max_val."""
    return max(min_val, min(max_val, value))

def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight

def format_response(analysis_result: Dict, user_query: str) -> str:
    """Format analysis result into readable response text."""
    if not analysis_result.get("success"):
        return f"Sorry, I couldn't analyze that request. Error: {analysis_result.get('error', 'Unknown error')}"
    
    market = analysis_result.get("market", {})
    ai_insights = analysis_result.get("ai_insights", {})
    
    coin_name = market.get("coin", "Unknown").capitalize()
    symbol = market.get("symbol", "N/A")
    price = market.get("price_usd", 0)
    change_24h = market.get("pct_change_24h", 0)
    
    recommendation = ai_insights.get("recommendation", "HOLD")
    confidence = ai_insights.get("confidence_score", 0.5)
    
    response = f"""
**{coin_name} ({symbol}) Analysis**

Current Price: {format_currency(price)}
24h Change: {format_percentage(change_24h)}

**Recommendation**: {recommendation}
**Confidence**: {confidence:.1%}

{ai_insights.get('detailed_analysis', 'Analysis completed successfully.')}

*This analysis was generated in response to: "{user_query}"*
    """
    
    return response.strip()

def create_error_response(error_message: str, user_query: str = "") -> Dict:
    """Create standardized error response."""
    return {
        "success": False,
        "error": error_message,
        "user_query": user_query,
        "timestamp": datetime.now().isoformat()
    }

def validate_api_response(response_data: Dict, required_fields: List[str]) -> bool:
    """Validate API response contains required fields."""
    if not response_data:
        return False
    return all(field in response_data for field in required_fields)

def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying functions on failure."""
    def wrapper(*args, **kwargs):
        import time
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
        
    return wrapper

def log_performance(func_name: str, duration: float, success: bool = True):
    """Log performance metrics for monitoring."""
    timestamp = datetime.now().isoformat()
    status = "SUCCESS" if success else "FAILURE"
    print(f"[{timestamp}] {func_name}: {duration:.3f}s - {status}")

# Data validation helpers
def is_valid_price(price: Union[int, float]) -> bool:
    """Check if price value is valid."""
    return isinstance(price, (int, float)) and price > 0 and not pd.isna(price)

def is_valid_percentage(pct: Union[int, float]) -> bool:
    """Check if percentage value is valid."""
    return isinstance(pct, (int, float)) and not pd.isna(pct) and -100 <= pct <= 1000

def sanitize_coin_id(coin_id: str) -> str:
    """Sanitize coin ID for safe usage."""
    if not coin_id:
        return "bitcoin"
    
    # Remove special characters and convert to lowercase
    sanitized = re.sub(r'[^a-zA-Z0-9\-]', '', coin_id.lower())
    
    # Ensure it's not empty
    return sanitized if sanitized else "bitcoin"
