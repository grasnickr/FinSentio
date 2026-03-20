# CNBC Sentiment Pipeline

An automated pipeline designed to extract and analyze financial news sentiment from CNBC. This project combines **reverse-engineered API access** with state-of-the-art **Natural Language Processing (NLP)** to quantify market sentiment for specific stock tickers.


---

## Features

* **Undocumented API Integration**: Efficiently fetches news data directly via CNBC's internal search infrastructure through reverse-engineered endpoints
* **Multi-Model Sentiment Analysis**: Supports **ProsusAI/FinBERT** (fine-tuned for financial texts) and **Flair** (general-purpose sentiment) — selectable via a simple `model` parameter
* **Model Comparison Tool**: Built-in script to compare FinBERT and Flair side-by-side on real CNBC news data
* **Automated Data Pipeline**: Generates cleaned Pandas DataFrames ready for downstream analysis, visualization, or integration with machine learning models
* **Hardware Optimized**: Supports CUDA acceleration for fast inference on NVIDIA GPUs (requires CUDA 12.8+ for RTX 50-series Blackwell GPUs)
* **Deduplication & Error Handling**: Robust duplicate detection and comprehensive error handling for reliable long-running operations

---

## Project Structure

```
CNBC-Sentiment-Pipeline/
├── StockSentiment.py          # Main module — API fetching, deduplication, pipeline orchestration
├── getsentimentFinBERT.py     # FinBERT sentiment scoring (financial-specific)
├── getsentimentFlair.py       # Flair sentiment scoring (general-purpose)
├── compare_models.py          # Side-by-side model comparison on real CNBC data
├── requirements.txt           # Python dependencies (PyTorch cu128, transformers, flair)
└── README.md
```

---

## Example Output

After fetching 500 articles for Micron (MU), the pipeline generates a DataFrame like this:


| Title | Published Date | FinBERT | Flair | URL |
| :--- | :--- | :--- | :--- | :--- |
| Micron gets an upgrade from Bank of America after blowout quarter | 2025-12-18 | +0.9370 | +0.8237 | [View Article](https://www.cnbc.com/2025/12/18/micron-gets-an-upgrade-from-bank-of-america-after-blowout-quarter-and-guidance.html) |
| Micron stock pops 10% as AI memory demand soars | 2025-12-18 | +0.4667 | -0.8773 | [View Article](https://www.cnbc.com/2025/12/18/micron-mu-stock-earnings-ai-memory-demand.html) |
| U.S. finalizes more than $6.1 billion chips subsidy for Micron | 2024-12-10 | +0.5885 | -0.9414 | [View Article](https://www.cnbc.com/2024/12/10/us-finalizes-more-than-6point1-billion-chips-subsidy-for-micron-technology.html) |
| Micron to exit server chips business in China after ban | 2025-10-17 | -0.0957 | -0.9996 | [View Article](https://www.cnbc.com/2025/10/17/micron-to-exit-server-chips-business-in-china-after-ban-report.html) |
| Micron shares suffer steepest drop since 2020 on weak guidance | 2024-12-19 | -0.9190 | -1.0000 | [View Article](https://www.cnbc.com/2024/12/19/micron-headed-for-worst-day-since-2020-after-disappointing-guidance.html) |

**Sentiment Score Range**: -1.0 (very negative) to +1.0 (very positive)

---

## Quick Start

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/grasnickr/CNBC-Sentiment-Pipeline
   cd CNBC-Sentiment-Pipeline
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```python
from StockSentiment import get_news_sentiment

# Fetch and analyze news using FinBERT (default)
ticker = "AAPL"
df = get_news_sentiment(maxpages=5, ticker=ticker)  # 5 pages = ~500 articles

# Or use Flair as an alternative sentiment model
df = get_news_sentiment(maxpages=5, ticker=ticker, model="flair")

# Display results
if not df.empty:
    print(f"\nAnalyzed {len(df)} articles for {ticker}")
    print(f"Average Sentiment: {df['sentiment_score'].mean():.4f}")
    print("\nMost Recent Articles:")
    print(df[['published_date', 'title', 'sentiment_score']].head(10))
```

### Model Comparison

Compare FinBERT and Flair side-by-side on real CNBC news:

```bash
# Compare models on 100 SPY articles (default)
python compare_models.py

# Compare on 200 Apple articles
python compare_models.py AAPL 2

# Compare on 100 Tesla articles
python compare_models.py TSLA 1
```

The comparison script outputs per-article scores, timing benchmarks, directional agreement, and the top 5 largest divergences between models.

### Advanced Example: Time Series Analysis

```python
import matplotlib.pyplot as plt

# Fetch data
df = get_news_sentiment(maxpages=10, ticker="TSLA")

# Aggregate daily sentiment
daily_sentiment = df.groupby(df['published_date'].dt.date)['sentiment_score'].mean()

# Plot sentiment over time
plt.figure(figsize=(12, 6))
plt.plot(daily_sentiment.index, daily_sentiment.values, marker='o')
plt.title('TSLA News Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Technical Details

### API Parameters
- **Base URL**: CNBC's Queryly search endpoint
- **Batch Size**: 100 articles per request
- **Pagination**: Automatic handling via `endindex` parameter
- **Rate Limiting**: Built-in request timeouts (20s) and error handling

### Sentiment Models

| Model | Best For | Speed | Behavior |
|---|---|---|---|
| `finbert` (default) | Financial texts | ~9ms/article | Nuanced scores, understands financial jargon like "golden cross", "downgrade", "guidance" |
| `flair` | General sentiment | ~5ms/article | Faster, tends toward extreme scores (close to +/-1.0) |

**Details:**
- **FinBERT**: [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) — BERT fine-tuned on financial data. Score = `P(positive) - P(negative)`
- **Flair**: Pre-trained DistilBERT sentiment classifier. Outputs signed confidence score

Both models:
- **Input**: Concatenated title + description (max 512 tokens for FinBERT)
- **Output**: Score range [-1.0, 1.0]
- **GPU**: CUDA acceleration when available

### GPU Requirements

- **CUDA 12.8+** required for NVIDIA RTX 50-series (Blackwell architecture)
- Older GPUs work with CUDA 12.6+
- Automatic CPU fallback if no GPU is available

---

## Use Cases

- **Quantitative Trading**: Incorporate sentiment signals into trading algorithms
- **Market Research**: Analyze media coverage trends for specific stocks or sectors
- **Academic Research**: Study correlation between news sentiment and stock price movements
- **Data Science Projects**: Use as a data source for ML models (e.g., LSTM price prediction)
- **Portfolio Analysis**: Monitor sentiment for entire portfolios

---

## Limitations & Considerations

- **API Stability**: Uses reverse-engineered endpoints that may change without notice
- **Rate Limiting**: No official rate limits known, but implement delays for large-scale scraping
- **Legal**: Ensure compliance with CNBC's Terms of Service for your use case
- **Sentiment Accuracy**: FinBERT is optimized for financial texts; Flair is general-purpose — use the comparison tool to evaluate for your use case
- **Historical Data**: API provides recent news; older articles may have limited availability

---
