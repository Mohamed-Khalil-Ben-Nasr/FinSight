"""Scrape Yahoo Finance and refresh the cached datasets."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finsight.pipeline import (
    DEFAULT_DATASET_PATH,
    DEFAULT_HISTORY_URL,
    compute_metrics_from_daily_records,
    load_history_dataset,
)


def main() -> None:
    dataset, note = load_history_dataset(
        dataset_path=DEFAULT_DATASET_PATH,
        dataset_url=DEFAULT_HISTORY_URL,
        refresh=True,
    )
    print(note)

    metrics = compute_metrics_from_daily_records(dataset)
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
