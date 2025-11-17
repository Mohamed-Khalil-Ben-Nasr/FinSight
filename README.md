# FinSight

FinSight is a multi-agent AI workflow that autonomously collects, analyzes, and forecasts stock market performance using the S&P 500 index.

## Running the pipeline
```bash
# Optional: install pandas/yfinance if you want to pull live data
pip install -r requirements.txt

# Run the end-to-end workflow (works even without optional deps)
python -m finsight.pipeline
```

If `pandas`/`yfinance` are unavailable or the network blocks downloads, the pipeline automatically falls back to a cached JSON snapshot containing the 2015–2022 annual S&P 500 returns and volatility, ensuring the Test Titan checks always run.

When execution finishes you’ll find:
- `artifacts/predictions_vs_actual.svg` – a quick chart comparing the Prediction Agent’s calls with reality.
- Console output from every agent, including Test Titan’s health check.
