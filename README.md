# AI Financial Analyst & Predictive Trading System

## Overview
A full-stack AI-powered financial trading system combining real-time and historical stock data, technical indicators, news sentiment, and advanced ML models to generate actionable trade signals and insights. UI inspired by Binomo.

## Features
- Real-time & historical OHLCV data
- Technical indicator computation (MACD, RSI, Bollinger Bands, etc.)
- News sentiment analysis (FinBERT/TextBlob)
- ML-based trade prediction (Buy/Sell/Hold, confidence, entry/exit)
- Top 5 stock picks with confidence
- Risk management & compliance filters
- REST API (FastAPI)

## Setup
1. Clone repo
2. `cd backend`
3. Install dependencies: `pip install -r ../requirements.txt`
4. Run API: `uvicorn backend.main:app --reload`

## Endpoints
- `/predict` (POST): Predict trade outcome for symbol
- `/top-picks` (GET): Get top 5 stocks likely to rise

## Notes
- For news sentiment, set your NewsAPI key in `backend/data.py` or use your preferred provider.
- This system is for informational purposes only. Not financial advice.

## Next Steps
- Add frontend (React, Binomo-style UI)
- Integrate advanced ML models (LSTM, XGBoost, FinBERT)
- Expand risk/compliance logic
