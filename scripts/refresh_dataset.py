"""Refresh the cached S&P 500 annual metrics snapshot using yfinance/pandas."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from finsight.pipeline import _compute_metrics_from_history


def main() -> None:
    ticker = yf.Ticker("^GSPC")
    history = ticker.history(start="2015-01-01", end="2022-12-31")
    if history.empty:
        raise RuntimeError("yfinance returned an empty frame; check the ticker or date range")

    metrics = _compute_metrics_from_history(history)
    payload = [
        {"Year": metric.year, "Avg_Return": metric.avg_return, "Volatility": metric.volatility}
        for metric in metrics
    ]

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "sp500_annual_metrics_2015_2022.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(payload)} annual records to {out_path}")


if __name__ == "__main__":
    main()
