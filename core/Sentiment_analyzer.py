"""
Sentiment analysis module using FinBERT for financial text analysis
"""

import pandas as pd
from typing import List, Dict, Tuple
from transformers import pipeline

class SentimentAnalyzer:
    """Handles sentiment analysis of news articles and headlines."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = -1):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name for sentiment analysis
            device: Device for inference (-1 for CPU, 0 for GPU)
        """
        self.model_name = model_name
        self.device = device
        self._pipeline = None
    
    def _load_pipeline(self):
        """Lazy loading of the sentiment analysis pipeline."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device
            )
        return self._pipeline
    
    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment of news articles.
        
        Args:
            articles: List of article dictionaries with 'title' keys
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not articles:
            return self._empty_sentiment_result()
        
        # Extract headlines for analysis
        headlines = [article.get("title", "") for article in articles if article.get("title")]
        
        if not headlines:
            return self._empty_sentiment_result()
        
        try:
            # Run sentiment analysis
            pipeline_obj = self._load_pipeline()
            predictions = pipeline_obj(headlines, truncation=True, max_length=512)
            
            # Process results
            sentiment_data = []
            for headline, prediction in zip(headlines, predictions):
                label = prediction.get("label", "").lower()
                confidence = float(prediction.get("score", 0.0))
                
                sentiment_data.append({
                    "text": headline,
                    "label": label,
                    "confidence": confidence,
                    "sentiment_value": self._label_to_value(label)
                })
            
            # Calculate aggregate metrics
            aggregate_score = self._calculate_aggregate_score(sentiment_data)
            percentages = self._calculate_percentages(sentiment_data)
            
            # Create results DataFrame
            sentiment_df = pd.DataFrame(sentiment_data)
            
            return {
                "sentiment_data": sentiment_data,
                "sentiment_dataframe": sentiment_df,
                "aggregate_score": aggregate_score,
                "percentages": percentages,
                "total_articles": len(sentiment_data)
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return self._empty_sentiment_result()
    
    def _label_to_value(self, label: str) -> float:
        """Convert sentiment label to numerical value."""
        mapping = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0
        }
        return mapping.get(label.lower(), 0.0)
    
    def _calculate_aggregate_score(self, sentiment_data: List[Dict]) -> float:
        """
        Calculate weighted aggregate sentiment score.
        
        Args:
            sentiment_data: List of sentiment analysis results
            
        Returns:
            Aggregate score between -1 and 1
        """
        if not sentiment_data:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for item in sentiment_data:
            value = item["sentiment_value"]
            weight = item["confidence"]
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_percentages(self, sentiment_data: List[Dict]) -> Dict[str, float]:
        """Calculate percentage distribution of sentiment labels."""
        if not sentiment_data:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
        total = len(sentiment_data)
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for item in sentiment_data:
            label = item["label"]
            if label in counts:
                counts[label] += 1
        
        return {
            label: (count / total) * 100.0 
            for label, count in counts.items()
        }
    
    def _empty_sentiment_result(self) -> Dict:
        """Return empty sentiment analysis result."""
        return {
            "sentiment_data": [],
            "sentiment_dataframe": pd.DataFrame(columns=["text", "label", "confidence", "sentiment_value"]),
            "aggregate_score": 0.0,
            "percentages": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
            "total_articles": 0
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        if not text.strip():
            return {"label": "neutral", "confidence": 0.0, "sentiment_value": 0.0}
        
        try:
            pipeline_obj = self._load_pipeline()
            result = pipeline_obj(text, truncation=True, max_length=512)[0]
            
            return {
                "label": result.get("label", "").lower(),
                "confidence": float(result.get("score", 0.0)),
                "sentiment_value": self._label_to_value(result.get("label", "")),
                "text": text
            }
            
        except Exception as e:
            print(f"Error analyzing text sentiment: {e}")
            return {"label": "neutral", "confidence": 0.0, "sentiment_value": 0.0}
    
    def get_sentiment_summary(self, sentiment_result: Dict) -> str:
        """
        Generate human-readable sentiment summary.
        
        Args:
            sentiment_result: Result from analyze_articles()
            
        Returns:
            Formatted sentiment summary string
        """
        if not sentiment_result or sentiment_result["total_articles"] == 0:
            return "No sentiment data available."
        
        score = sentiment_result["aggregate_score"]
        percentages = sentiment_result["percentages"]
        total = sentiment_result["total_articles"]
        
        # Determine overall sentiment