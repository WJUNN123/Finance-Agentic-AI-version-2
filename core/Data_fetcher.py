"""
Data fetching module for cryptocurrency market data and news
"""

import requests
import pandas as pd
import numpy as np
import feedparser
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

class DataFetcher:
    """Handles fetching market data and news articles."""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.request_timeout = 20
        
    def get_market_data(self, coin_ids: List[str]) -> pd.DataFrame:
        """
        Fetch current market data for specified coins from CoinGecko.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            
        Returns:
            DataFrame with market data
        """
        if not coin_ids:
            return pd.DataFrame()
            
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": len(coin_ids),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def get_price_history(self, coin_id: str, days: int = 180) -> pd.DataFrame:
        """
        Fetch historical price data for a coin.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of history to fetch
            
        Returns:
            DataFrame with timestamp index and price column
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        
        try:
            response = requests.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            
            prices = data.get("prices", [])
            if not prices:
                return pd.DataFrame(columns=["price"])
                
            df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df.drop(columns=["timestamp_ms"], inplace=True)
            
            return df
            
        except requests.RequestException as e:
            print(f"Error fetching price history: {e}")
            return pd.DataFrame(columns=["price"])
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of price data
            period: RSI period (default 14)
            
        Returns:
            RSI value (0-100) or NaN if insufficient data
        """
        if len(prices) < period + 1:
            return float('nan')
            
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else float('nan')
    
    def get_news_articles(self, coin_symbol: str, coin_name: str, 
                         limit_per_feed: int = 20) -> List[Dict]:
        """
        Fetch news articles related to a cryptocurrency.
        
        Args:
            coin_symbol: Coin symbol (e.g., 'BTC')
            coin_name: Coin name (e.g., 'bitcoin')
            limit_per_feed: Maximum articles per RSS feed
            
        Returns:
            List of article dictionaries
        """
        rss_feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://news.google.com/rss/search?q=cryptocurrency&hl=en-US&gl=US&ceid=US:en",
        ]
        
        search_terms = [coin_symbol.lower(), coin_name.lower()]
        articles = []
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:limit_per_feed]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    link = entry.get("link", "")
                    
                    # Check if article is relevant
                    content = f"{title} {summary}".lower()
                    if any(term in content for term in search_terms):
                        
                        # Parse publication date
                        published = entry.get("published_parsed") or entry.get("updated_parsed")
                        pub_timestamp = time.mktime(published) if published else time.time()
                        
                        articles.append({
                            "title": title,
                            "summary": summary,
                            "link": link,
                            "published_timestamp": pub_timestamp,
                            "published_date": self._format_timestamp(pub_timestamp),
                            "source": feed_url,
                        })
                        
            except Exception as e:
                print(f"Error fetching from {feed_url}: {e}")
                continue
        
        # Remove duplicates and sort by date
        seen_titles = set()
        unique_articles = []
        
        for article in sorted(articles, key=lambda x: x["published_timestamp"], reverse=True):
            if article["title"] not in seen_titles:
                seen_titles.add(article["title"])
                unique_articles.append(article)
        
        return unique_articles[:50]  # Return top 50 articles
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp to human-readable date."""
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except:
            return "Unknown"
    
    def get_market_metrics(self, price_history: pd.DataFrame) -> Dict:
        """
        Calculate additional market metrics from price history.
        
        Args:
            price_history: DataFrame with price data
            
        Returns:
            Dictionary of calculated metrics
        """
        if price_history.empty or "price" not in price_history.columns:
            return {}
        
        prices = price_history["price"]
        
        metrics = {
            "rsi_14": self.calculate_rsi(prices, 14),
            "ma_7": prices.rolling(7).mean().iloc[-1] if len(prices) >= 7 else None,
            "ma_14": prices.rolling(14).mean().iloc[-1] if len(prices) >= 14 else None,
            "ma_30": prices.rolling(30).mean().iloc[-1] if len(prices) >= 30 else None,
        }
        
        # Calculate volatility (standard deviation of returns)
        if len(prices) > 1:
            returns = prices.pct_change().dropna()
            metrics["volatility_7d"] = returns.tail(7).std() if len(returns) >= 7 else None
            metrics["volatility_30d"] = returns.tail(30).std() if len(returns) >= 30 else None
        
        return {k: v for k, v in metrics.items() if v is not None}
