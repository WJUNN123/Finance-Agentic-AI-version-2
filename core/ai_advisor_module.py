"""
AI advisor module for generating cryptocurrency investment insights using Google Gemini
"""

import math
import re
from typing import Dict, List, Optional
import google.generativeai as genai
from .Config import AppConfig

class AIAdvisor:
    """Handles AI-powered investment insights and recommendations."""
    
    def __init__(self):
        self.config = AppConfig()
        self.api_key = self.config.get_api_key("gemini")
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model with API key."""
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"Failed to initialize Gemini model: {e}")
                self.model = None
        else:
            print("No Gemini API key found. AI insights will use fallback analysis.")
    
    def generate_insights(self, coin_id: str, coin_symbol: str, 
                         market_data: Dict, sentiment_data: Dict,
                         forecast_data: Dict, horizon_days: int = 7) -> Dict:
        """
        Generate comprehensive investment insights using AI.
        
        Args:
            coin_id: Cryptocurrency ID
            coin_symbol: Cryptocurrency symbol
            market_data: Current market data
            sentiment_data: Sentiment analysis results
            forecast_data: Price forecast results
            horizon_days: Investment horizon in days
            
        Returns:
            Dictionary containing AI insights and recommendations
        """
        if self.model:
            return self._generate_ai_insights(
                coin_id, coin_symbol, market_data, 
                sentiment_data, forecast_data, horizon_days
            )
        else:
            return self._generate_fallback_insights(
                coin_id, coin_symbol, market_data,
                sentiment_data, forecast_data, horizon_days
            )
    
    def _generate_ai_insights(self, coin_id: str, coin_symbol: str,
                             market_data: Dict, sentiment_data: Dict,
                             forecast_data: Dict, horizon_days: int) -> Dict:
        """Generate insights using Gemini AI."""
        try:
            prompt = self._build_analysis_prompt(
                coin_id, coin_symbol, market_data,
                sentiment_data, forecast_data, horizon_days
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=800,
                    top_p=0.9,
                    top_k=40
                )
            )
            
            ai_text = response.text.strip()
            
            # Extract structured data from AI response
            recommendation = self._extract_recommendation(ai_text)
            confidence_score = self._calculate_confidence_score(
                market_data, sentiment_data, forecast_data
            )
            
            return {
                'recommendation': recommendation,
                'confidence_score': confidence_score,
                'detailed_analysis': ai_text,
                'risk_assessment': self._extract_risk_assessment(ai_text),
                'key_factors': self._extract_key_factors(ai_text),
                'source': 'gemini',
                'success': True
            }
            
        except Exception as e:
            print(f"AI insight generation failed: {e}")
            return self._generate_fallback_insights(
                coin_id, coin_symbol, market_data,
                sentiment_data, forecast_data, horizon_days
            )
    
    def _build_analysis_prompt(self, coin_id: str, coin_symbol: str,
                              market_data: Dict, sentiment_data: Dict,
                              forecast_data: Dict, horizon_days: int) -> str:
        """Build comprehensive analysis prompt for AI."""
        
        # Format market data safely
        def safe_format(value, default="N/A", format_str="{:.2f}"):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            return format_str.format(value)
        
        price = market_data.get('price_usd', 0)
        change_24h = market_data.get('pct_change_24h')
        change_7d = market_data.get('pct_change_7d')
        market_cap = market_data.get('market_cap', 0)
        volume = market_data.get('volume_24h', 0)
        rsi = market_data.get('rsi_14')
        
        # Sentiment data
        sentiment_score = sentiment_data.get('aggregate_score', 0)
        sentiment_pcts = sentiment_data.get('percentages', {})
        
        # Forecast data
        forecast_summary = ""
        if forecast_data.get('ensemble_forecast'):
            forecast_values = forecast_data['ensemble_forecast']['values']
            if forecast_values:
                expected_change = ((forecast_values[-1] - price) / price * 100) if price > 0 else 0
                forecast_summary = f"7-day forecast suggests {expected_change:+.1f}% price movement"
        
        prompt = f"""
You are an expert cryptocurrency analyst. Analyze {coin_symbol.upper()} ({coin_id}) and provide investment insights.

CURRENT MARKET DATA:
- Price: ${safe_format(price)}
- Market Cap: ${safe_format(market_cap, format_str="{:,.0f}")}
- 24h Volume: ${safe_format(volume, format_str="{:,.0f}")}
- 24h Change: {safe_format(change_24h)}%
- 7d Change: {safe_format(change_7d)}%
- RSI(14): {safe_format(rsi)} ({self._get_rsi_zone(rsi)})

SENTIMENT ANALYSIS:
- Overall Sentiment Score: {sentiment_score:.3f} (range: -1 to +1)
- Positive: {sentiment_pcts.get('positive', 0):.1f}%
- Neutral: {sentiment_pcts.get('neutral', 0):.1f}%
- Negative: {sentiment_pcts.get('negative', 0):.1f}%

FORECAST: {forecast_summary}

ANALYSIS PARAMETERS:
- Investment Horizon: {horizon_days} days
- Analysis Date: Today

Please provide:

1. **RECOMMENDATION**: Clear BUY/SELL/HOLD recommendation with rationale

2. **TECHNICAL ANALYSIS**: 
   - Price momentum assessment (24h and 7d trends)
   - RSI analysis and overbought/oversold conditions
   - Support and resistance levels if identifiable

3. **SENTIMENT IMPACT**:
   - How current news sentiment affects the outlook
   - Key narrative themes affecting the cryptocurrency

4. **RISK FACTORS**:
   - Primary risks to consider for this investment
   - Market conditions that could change the outlook

5. **STRATEGIC CONSIDERATIONS**:
   - Entry/exit points if applicable
   - Position sizing recommendations
   - Key metrics to monitor

Format your response clearly with sections. Be specific about price levels and timeframes where possible. Include appropriate disclaimers about investment risk.
        """
        
        return prompt
    
    def _generate_fallback_insights(self, coin_id: str, coin_symbol: str,
                                   market_data: Dict, sentiment_data: Dict,
                                   forecast_data: Dict, horizon_days: int) -> Dict:
        """Generate rule-based insights when AI is unavailable."""
        
        price = market_data.get('price_usd', 0)
        change_24h = market_data.get('pct_change_24h', 0)
        change_7d = market_data.get('pct_change_7d', 0)
        rsi = market_data.get('rsi_14', 50)
        sentiment_score = sentiment_data.get('aggregate_score', 0)
        
        # Rule-based recommendation logic
        recommendation = self._rule_based_recommendation(
            change_24h, change_7d, rsi, sentiment_score
        )
        
        # Calculate confidence based on signal strength
        confidence_score = self._calculate_confidence_score(
            market_data, sentiment_data, forecast_data
        )
        
        # Generate analysis text
        analysis_text = self._generate_fallback_analysis_text(
            coin_symbol, market_data, sentiment_data, recommendation
        )
        
        return {
            'recommendation': recommendation,
            'confidence_score': confidence_score,
            'detailed_analysis': analysis_text,
            'risk_assessment': self._generate_risk_assessment(market_data),
            'key_factors': self._identify_key_factors(market_data, sentiment_data),
            'source': 'rule_based',
            'success': True
        }
    
    def _rule_based_recommendation(self, change_24h: float, change_7d: float,
                                  rsi: float, sentiment: float) -> str:
        """Generate recommendation using rule-based logic."""
        
        # Strong buy signals
        if (sentiment > 0.3 and change_7d > 5 and rsi < 30):
            return "STRONG BUY"
        elif (sentiment > 0.1 and change_24h > 2 and rsi < 40):
            return "BUY"
        
        # Strong sell signals
        elif (sentiment < -0.3 and change_7d < -10 and rsi > 70):
            return "STRONG SELL"
        elif (sentiment < -0.1 and change_24h < -5 and rsi > 60):
            return "SELL"
        
        # Hold conditions
        elif abs(sentiment) < 0.1 and abs(change_24h) < 3:
            return "HOLD"
        
        # Default to cautious hold
        else:
            return "HOLD"
    
    def _calculate_confidence_score(self, market_data: Dict, 
                                   sentiment_data: Dict, 
                                   forecast_data: Dict) -> float:
        """Calculate confidence score for the recommendation."""
        
        score_components = []
        
        # Sentiment strength
        sentiment_score = abs(sentiment_data.get('aggregate_score', 0))
        score_components.append(min(sentiment_score, 1.0))
        
        # Price momentum consistency
        change_24h = market_data.get('pct_change_24h', 0)
        change_7d = market_data.get('pct_change_7d', 0)
        
        if change_24h and change_7d:
            momentum_consistency = 1.0 if (change_24h > 0) == (change_7d > 0) else 0.5
            score_components.append(momentum_consistency)
        
        # Volume validation (if available)
        volume = market_data.get('volume_24h', 0)
        market_cap = market_data.get('market_cap', 1)
        if volume and market_cap:
            volume_ratio = min(volume / market_cap, 0.1) * 10  # Normalize to 0-1
            score_components.append(volume_ratio)
        
        # Average the components
        return sum(score_components) / len(score_components) if score_components else 0.5
    
    def _extract_recommendation(self, ai_text: str) -> str:
        """Extract recommendation from AI response."""
        ai_lower = ai_text.lower()
        
        if any(phrase in ai_lower for phrase in ["strong buy", "strongly recommend buying"]):
            return "STRONG BUY"
        elif "buy" in ai_lower and "don't buy" not in ai_lower:
            return "BUY"
        elif any(phrase in ai_lower for phrase in ["strong sell", "sell immediately"]):
            return "STRONG SELL"
        elif "sell" in ai_lower:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_risk_assessment(self, ai_text: str) -> List[str]:
        """Extract risk factors from AI response."""
        risks = []
        risk_keywords = [
            "volatility", "regulatory", "liquidity", "market risk",
            "technical risk", "adoption risk", "competition"
        ]
        
        sentences = ai_text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in risk_keywords):
                if "risk" in sentence_lower:
                    risks.append(sentence.strip())
        
        return risks[:3]  # Return top 3 risk factors
    
    def _extract_key_factors(self, ai_text: str) -> List[str]:
        """Extract key factors from AI response."""
        factors = []
        factor_keywords = [
            "momentum", "sentiment", "support", "resistance",
            "trend", "volume", "adoption", "development"
        ]
        
        sentences = ai_text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in factor_keywords):
                factors.append(sentence.strip())
        
        return factors[:4]  # Return top 4 factors
    
    def _get_rsi_zone(self, rsi: float) -> str:
        """Get RSI zone description."""
        if rsi is None or math.isnan(rsi):
            return "Unknown"
        elif rsi >= 70:
            return "Overbought"
        elif rsi <= 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def _generate_fallback_analysis_text(self, coin_symbol: str, 
                                        market_data: Dict,
                                        sentiment_data: Dict, 
                                        recommendation: str) -> str:
        """Generate analysis text for fallback mode."""
        
        change_24h = market_data.get('pct_change_24h', 0)
        change_7d = market_data.get('pct_change_7d', 0)
        sentiment_score = sentiment_data.get('aggregate_score', 0)
        
        analysis = f"""
**RECOMMENDATION**: {recommendation}

**TECHNICAL ANALYSIS**:
{coin_symbol} shows 24h change of {change_24h:+.2f}% and 7-day change of {change_7d:+.2f}%. 
The momentum is {'positive' if change_24h > 0 else 'negative'} in the short term.

**SENTIMENT IMPACT**:
News sentiment score is {sentiment_score:+.3f}, indicating 
{'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'} 
market sentiment based on recent headlines.

**RISK FACTORS**:
- Cryptocurrency investments carry high volatility risk
- Market sentiment can change rapidly
- Regulatory developments may impact prices

**Note**: This analysis is generated using rule-based logic. 
For more sophisticated insights, ensure AI service is properly configured.
        """
        
        return analysis.strip()
    
    def _generate_risk_assessment(self, market_data: Dict) -> List[str]:
        """Generate risk assessment for fallback mode."""
        risks = ["High volatility typical of cryptocurrency markets"]
        
        volume = market_data.get('volume_24h', 0)
        market_cap = market_data.get('market_cap', 1)
        
        if volume and market_cap:
            liquidity_ratio = volume / market_cap
            if liquidity_ratio < 0.01:
                risks.append("Low liquidity may cause price slippage")
        
        risks.append("Regulatory changes could impact market dynamics")
        
        return risks
    
    def _identify_key_factors(self, market_data: Dict, 
                             sentiment_data: Dict) -> List[str]:
        """Identify key factors for fallback mode."""
        factors = []
        
        change_24h = market_data.get('pct_change_24h', 0)
        if abs(change_24h) > 5:
            factors.append(f"Significant 24h price movement: {change_24h:+.2f}%")
        
        sentiment_score = sentiment_data.get('aggregate_score', 0)
        if abs(sentiment_score) > 0.2:
            sentiment_desc = "positive" if sentiment_score > 0 else "negative"
            factors.append(f"Strong {sentiment_desc} news sentiment")
        
        factors.append("Overall market conditions and crypto sector trends")
        
        return factors
