# 📈 NEV-Sentiment-Intelligence

**NEV-Sentiment-Intelligence** is a deep-learning analytics platform designed for the New Energy Vehicle (NEV) market. It features a robust pipeline for sentiment trend forecasting, anomaly detection, and competitive intelligence using **LSTM** neural networks and weighted engagement metrics.

---

### Key Features
- **LSTM-Driven Forecasting**: Advanced time-series prediction (optimized over deprecated Linear Regression, Random Forest, and ARIMA models).
- **Dual-Granularity Analysis**: Supports flexible sentiment tracking across both **Brand** (e.g., Tesla, NIO) and **Keyword** (e.g., Battery, Autopilot) dimensions.
- **Weighted Sentiment Scoring**: Incorporates social engagement (Likes, Shares) into polarity calculation: `Score = 1 + α*likes + β*shares`.
- **Statistical Suite**: Integrated seasonal decomposition, 95% confidence intervals, and automated anomaly detection.
- **Interactive Dashboards**: Generates dynamic HTML trend charts and sentiment distribution pies.

---

### Use Cases
- **Public Opinion Forecasting**: Predicting sentiment shifts for the next 30 days to support strategic planning.
- **Brand Reputation Monitoring**: Tracking long-term health and perception of NEV manufacturers.
- **Keyword-Level Deep Dives**: Identifying specific technical or service pain points (e.g., charging speed vs. interior quality).
- **PR Crisis Early Warning**: Triggering alerts when negative sentiment spikes or exceeds predefined thresholds.

---

### Models & Tech Stack

#### Core Forecasting Model
- **LSTM (PyTorch)**: A many-to-one recurrent architecture optimized for capturing non-linear sentiment patterns. 
- *Note: This system has evolved to use LSTM exclusively, deprecating traditional statistical methods for higher accuracy.*

#### Core Libraries
- **Data Science**: `Pandas`, `NumPy`, `Statsmodels` (Seasonal Decomposition).
- **Visualization**: `Plotly` (Interactive Web Renderer), `Matplotlib`.
- **NLP**: `Transformers` (BERT-based features), `SnowNLP` (Fallback analyzer).
- **GUI**: `Tkinter` (For seamless file and brand/keyword selection).

---

### 📂 Project Structure
```text
NEV-Sentiment-Intelligence/
├── sentiment_trend.py       # Main entry: Forecast engine & Analysis logic
├── config.py                # Configuration for weights, thresholds, and APIs
├── analyzer/                # Modules for Entity Recognition & Attack Analysis
├── output/                  # Auto-generated HTML charts and TXT reports
├── mock_data.csv            # Sample input dataset
└── requirements.txt         # Project dependencies
```