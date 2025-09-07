# Crypto Analysis Dashboard - Environment Variables
# Copy this file to .env and add your actual API keys

# Google Gemini API Key (Required)
# Get your key from: https://makersuite.google.com/
GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face Token (Optional)
# Needed only for private models or higher rate limits
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here

# OpenAI API Key (Optional - for alternative AI insights)
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration (Optional)
DATABASE_URL=sqlite:///data/crypto_analysis.db

# Caching Configuration (Optional)
CACHE_DURATION_MARKET_DATA=300  # 5 minutes in seconds
CACHE_DURATION_NEWS=900         # 15 minutes in seconds
CACHE_DURATION_FORECASTS=3600   # 1 hour in seconds

# Model Configuration (Optional)
FINBERT_MODEL=ProsusAI/finbert
SENTIMENT_DEVICE=-1  # -1 for CPU, 0 for GPU

# Forecast Parameters (Optional)
LSTM_WINDOW=30
LSTM_EPOCHS=20
LSTM_BATCH_SIZE=16

# Debug Mode (Optional)
DEBUG=False
LOG_LEVEL=INFO