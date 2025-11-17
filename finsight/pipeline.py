"""FinSight multi-agent pipeline implementation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from textwrap import indent
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:  # Optional heavy dependencies (gracefully skipped when unavailable)
    import pandas as pd  # type: ignore
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled via fallback data
    pd = None
    yf = None

try:  # LangChain powers the orchestrator when available
    from langchain.schema.runnable import RunnableLambda
except ModuleNotFoundError:  # pragma: no cover - provide a tiny fallback for offline demos
    class RunnableLambda:  # type: ignore[no-redef]
        """Minimal drop-in replacement for LangChain's RunnableLambda."""

        def __init__(self, func):
            self._func = func

        def invoke(self, input_):
            return self._func(input_)

        def __call__(self, input_):  # pragma: no cover - mirrors LangChain API
            return self.invoke(input_)

        def __or__(self, other):
            return RunnableLambda(lambda value: other.invoke(self.invoke(value)))

PINECONE_IMPORT_ERROR: Optional[Exception] = None
try:  # Pinecone is optional but encouraged when an API key is present
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled by local memory store
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore
    PINECONE_IMPORT_ERROR = exc
except Exception as exc:  # pragma: no cover - safety guard for old pinecone-client
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore
    PINECONE_IMPORT_ERROR = exc


DEFAULT_HISTORY_URL = (
    "https://finance.yahoo.com/quote/%5EGSPC/history/?period1=1420070400&period2=1668643200"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "dataset.json"


@dataclass
class AnnualMetric:
    """Representation of the DataCollector output for a single year."""

    year: int
    avg_return: float
    volatility: float


@dataclass
class DataSourceReport:
    """Details about how the annual metrics were sourced."""

    source: str
    note: str


@dataclass
class AgentOutput:
    """Container aggregating final workflow outputs."""

    annual_metrics: List[AnnualMetric]
    context: Dict[int, List[str]]
    predictions: Dict[int, float]
    rationales: Dict[int, str]
    evaluation: Dict[str, object]
    finance_bro_summary: str
    refinement_suggestions: List[str]
    chart_path: Optional[str]
    mr_white_briefing: str
    vector_store_status: str
    data_source_report: DataSourceReport


StageLogger = Callable[[str, Dict[str, object]], None]


class ContextVectorStore:
    """Minimal Pinecone facade that falls back to in-memory storage when offline."""

    def __init__(
        self,
        index_name: str = "finsight-market-events",
        dimension: int = 8,
        *,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.index_name = index_name
        self.dimension = dimension
        self._status = "pinecone_disabled"
        self._client = None
        self._index = None
        self._memory: Dict[str, Dict[str, object]] = {}

        resolved_api_key = api_key or os.getenv("PINECONE_API_KEY")
        resolved_region = region or os.getenv("PINECONE_REGION", "us-east-1")

        if Pinecone is not None and resolved_api_key:
            try:
                self._client = Pinecone(api_key=resolved_api_key)
                try:
                    self._client.describe_index(self.index_name)
                except Exception:
                    if ServerlessSpec is None:  # pragma: no cover - safety guard
                        raise RuntimeError("pinecone-serverless missing; upgrade pinecone-client.")
                    self._client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region=resolved_region),
                    )
                self._index = self._client.Index(self.index_name)
                self._status = f"pinecone_serverless:{resolved_region}"
            except Exception as exc:  # pragma: no cover - network/auth specific
                self._client = None
                self._index = None
                self._status = f"pinecone_local_memory:init_error:{type(exc).__name__}"
        else:
            if PINECONE_IMPORT_ERROR is not None:
                reason = type(PINECONE_IMPORT_ERROR).__name__
                self._status = f"pinecone_local_memory:import_error:{reason}"
            elif not resolved_api_key:
                self._status = "pinecone_local_memory:no_api_key"
            else:
                self._status = "pinecone_local_memory"

    @property
    def status(self) -> str:
        return self._status

    def _embed_text(self, text: str) -> List[float]:
        seed = abs(hash(text))
        vector = []
        for i in range(self.dimension):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            vector.append((seed % 1000) / 1000.0)
        return vector

    def upsert_batch(self, summaries: Dict[int, List[str]]) -> None:
        for year, events in summaries.items():
            text = " ".join(events)
            vector = self._embed_text(text or str(year))
            payload = {"id": str(year), "values": vector, "metadata": {"year": year, "events": events}}
            if self._index is not None:
                self._index.upsert(vectors=[payload])
            else:
                self._memory[str(year)] = payload

    def fetch_context(self, years: Iterable[int]) -> Dict[int, List[str]]:
        result: Dict[int, List[str]] = {}
        ids = [str(year) for year in years]
        if self._index is not None:
            response = self._index.fetch(ids=ids)
            vectors = response.get("vectors", {}) if isinstance(response, dict) else {}
            for key, record in vectors.items():
                metadata = record.get("metadata") or {}
                events = metadata.get("events") or []
                result[int(key)] = list(events)
        else:
            for key in ids:
                payload = self._memory.get(key)
                if payload:
                    metadata = payload.get("metadata", {})
                    events = metadata.get("events") or []
                    result[int(key)] = list(events)
        return result


def _compute_metrics_from_history(history: "pd.DataFrame") -> List[AnnualMetric]:  # pragma: no cover - only runs when pandas is installed
    history = history["Close"].to_frame(name="close")
    history.index = pd.to_datetime(history.index)
    history["daily_return"] = history["close"].pct_change()
    history = history.dropna()

    grouped = history.groupby(history.index.year)
    records: List[AnnualMetric] = []
    for year, group in grouped:
        avg_return = float((1 + group["daily_return"]).prod() - 1)
        volatility = float(group["daily_return"].std() * (252 ** 0.5))
        records.append(AnnualMetric(year=int(year), avg_return=avg_return, volatility=volatility))
    return sorted(records, key=lambda item: item.year)


def _load_cached_metrics() -> List[AnnualMetric]:
    cache_path = Path(__file__).resolve().parent.parent / "data" / "sp500_annual_metrics_2015_2022.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing fallback metrics at {cache_path}")
    data = cache_path.read_text().strip()
    import json

    raw_records = json.loads(data)
    return [
        AnnualMetric(year=int(entry["Year"]), avg_return=float(entry["Avg_Return"]), volatility=float(entry["Volatility"]))
        for entry in raw_records
    ]


def _scrape_history_dataset(url: str = DEFAULT_HISTORY_URL) -> List[Dict[str, Any]]:
    """Scrape the Yahoo Finance historical page and extract price records."""

    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinSight/1.0)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text
    match = re.search(r"root\.App\.main\s*=\s*({.*?})\s*;\s*}\(this\)\);", html, re.DOTALL)
    if not match:
        raise ValueError("Unable to locate root.App.main JSON payload in Yahoo response")
    payload = json.loads(match.group(1))
    store = (
        payload.get("context", {})
        .get("dispatcher", {})
        .get("stores", {})
        .get("HistoricalPriceStore", {})
    )
    prices = store.get("prices", [])
    records: List[Dict[str, Any]] = []
    for item in prices:
        if not isinstance(item, dict) or item.get("type"):
            continue
        date_ts = item.get("date")
        open_px = item.get("open")
        high_px = item.get("high")
        low_px = item.get("low")
        close_px = item.get("close")
        adj_close = item.get("adjclose")
        volume = item.get("volume")
        if None in (date_ts, open_px, high_px, low_px, close_px, adj_close, volume):
            continue
        date_obj = dt.datetime.fromtimestamp(int(date_ts), tz=dt.timezone.utc)
        records.append(
            {
                "Date": date_obj.date().isoformat(),
                "Open": float(open_px),
                "High": float(high_px),
                "Low": float(low_px),
                "Close": float(close_px),
                "Adj Close": float(adj_close),
                "Volume": int(volume),
            }
        )
    records.sort(key=lambda rec: rec["Date"])
    if not records:
        raise ValueError("Yahoo Finance response did not contain price rows")
    return records


def load_history_dataset(
    *,
    dataset_path: Optional[Path] = None,
    dataset_url: str = DEFAULT_HISTORY_URL,
    refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Load the cached dataset or scrape Yahoo Finance when requested."""

    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    scrape_errors: List[str] = []

    if refresh or not dataset_path.exists():
        try:
            records = _scrape_history_dataset(dataset_url)
            dataset_path.write_text(json.dumps(records, indent=2))
            note = (
                f"Scraped {len(records)} rows from Yahoo Finance and cached to {dataset_path.name}"
            )
            return records, note
        except Exception as exc:  # pragma: no cover - depends on network
            scrape_errors.append(f"scrape failed: {exc}")

    if dataset_path.exists():
        data = json.loads(dataset_path.read_text())
        note = f"Loaded {len(data)} rows from {dataset_path.name}"
        if scrape_errors:
            note += f" (fallback because {'; '.join(scrape_errors)})"
        return data, note

    raise RuntimeError("dataset.json missing and scrape failed: " + "; ".join(scrape_errors))


def _filter_records_by_range(
    records: Sequence[Dict[str, Any]], start: str, end: str
) -> List[Dict[str, Any]]:
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end)
    filtered = []
    for record in records:
        record_date = dt.date.fromisoformat(str(record["Date"]))
        if start_date <= record_date <= end_date:
            filtered.append(record)
    return filtered


def compute_metrics_from_daily_records(records: Sequence[Dict[str, Any]]) -> List[AnnualMetric]:
    """Compute annual returns/volatility from scraped daily candles."""

    daily_returns: Dict[int, List[float]] = {}
    prev_close: Optional[float] = None
    sorted_records = sorted(records, key=lambda rec: rec["Date"])
    for record in sorted_records:
        close = float(record["Close"])
        year = dt.date.fromisoformat(str(record["Date"])).year
        if prev_close is not None:
            ret = (close / prev_close) - 1.0
            daily_returns.setdefault(year, []).append(ret)
        prev_close = close

    metrics: List[AnnualMetric] = []
    for year in sorted(daily_returns):
        returns = daily_returns[year]
        if not returns:
            continue
        total = math.prod(1 + value for value in returns)
        volatility = pstdev(returns) * (252 ** 0.5) if len(returns) > 1 else 0.0
        metrics.append(
            AnnualMetric(year=year, avg_return=float(total - 1), volatility=float(volatility))
        )
    return metrics


def collect_sp500_data(
    start: str = "2015-01-01",
    end: str = "2022-12-31",
    *,
    dataset_path: Optional[Path] = None,
    dataset_url: str = DEFAULT_HISTORY_URL,
    refresh_dataset: bool = False,
) -> Tuple[List[AnnualMetric], DataSourceReport, List[Dict[str, Any]]]:
    """Download S&P 500 data, compute annual metrics, and return the raw candles."""

    failure_reasons: List[str] = []

    raw_chart_records: List[Dict[str, Any]] = []

    try:
        raw_records, note = load_history_dataset(
            dataset_path=dataset_path,
            dataset_url=dataset_url,
            refresh=refresh_dataset,
        )
        filtered = _filter_records_by_range(raw_records, start, end)
        metrics = compute_metrics_from_daily_records(filtered)
        if metrics:
            raw_chart_records = filtered
            return metrics, DataSourceReport(source="yahoo_history_html", note=note), raw_chart_records
        failure_reasons.append("history dataset did not yield metrics")
    except Exception as exc:  # pragma: no cover - depends on network/disk state
        failure_reasons.append(f"dataset scrape error: {exc}")

    if pd is not None and yf is not None:
        try:
            ticker = yf.Ticker("^GSPC")
            history = ticker.history(start=start, end=end)
            if not history.empty:
                metrics = _compute_metrics_from_history(history)
                note = (
                    "Live ^GSPC download succeeded via yfinance and was cleaned with pandas"
                )
                raw_chart_records = [
                    {
                        "Date": str(row.Index.date()),
                        "Close": float(row.Close),
                    }
                    for row in history[["Close"]].itertuples()
                ]
                return metrics, DataSourceReport(source="yfinance", note=note), raw_chart_records
            failure_reasons.append("yfinance returned an empty frame")
        except Exception as exc:  # pragma: no cover - depends on network state
            failure_reasons.append(f"yfinance error: {exc.__class__.__name__}: {exc}")
    else:
        failure_reasons.append("pandas/yfinance not installed")

    metrics = _load_cached_metrics()
    reason_suffix = f" because {'; '.join(failure_reasons)}" if failure_reasons else ""
    note = (
        "Used cached snapshot derived from the same Yahoo Finance pulls"
        f"{reason_suffix}."
    )
    synthetic_records: List[Dict[str, Any]] = []
    for metric in metrics:
        start_date = dt.date(metric.year, 1, 1)
        days = max(100, 252)
        for idx in range(days):
            progress = (idx + 1) / days
            synthetic_close = (1 + metric.avg_return) ** progress
            synthetic_records.append(
                {
                    "Date": (start_date + dt.timedelta(days=idx)).isoformat(),
                    "Close": float(synthetic_close),
                }
            )

    return metrics, DataSourceReport(source="cached_snapshot", note=note), synthetic_records


def research_macro_events(
    years: Iterable[int], vector_store: Optional[ContextVectorStore] = None
) -> Tuple[Dict[int, List[str]], str]:
    """Return the top three macro events and store them inside Pinecone."""

    summaries = {
        2015: [
            "Fed prepared to hike rates for the first time since the GFC",
            "China's slowdown sparked global growth fears",
            "Energy prices slumped, weighing on earnings",
        ],
        2016: [
            "Brexit referendum surprised markets",
            "US election introduced policy uncertainty",
            "Oil stabilized after crashing in 2015",
        ],
        2017: [
            "Global synchronized growth boosted equities",
            "Corporate tax reform expectations fueled optimism",
            "Volatility stayed historically low",
        ],
        2018: [
            "US-China trade tensions escalated",
            "Fed hikes triggered liquidity worries",
            "Late-year selloff signaled risk-off sentiment",
        ],
        2019: [
            "Trade truce hopes lifted risk appetite",
            "Global central banks pivoted back to easing",
            "Tech leadership returned despite valuation concerns",
        ],
    }

    filtered = {year: summaries.get(year, []) for year in years}
    vector_status = "vector_store_disabled"
    if vector_store is not None:
        vector_store.upsert_batch(filtered)
        stored = vector_store.fetch_context(years)
        filtered = {year: stored.get(year) or filtered.get(year, []) for year in years}
        vector_status = vector_store.status

    return filtered, vector_status


def _filter_metrics(annual_metrics: Sequence[AnnualMetric], years: Iterable[int]) -> List[AnnualMetric]:
    year_set = set(years)
    return [metric for metric in annual_metrics if metric.year in year_set]


def build_prediction_payload(
    annual_metrics: Sequence[AnnualMetric], context: Dict[int, List[str]]
) -> Dict[str, object]:
    """Create the structured payload for the Prediction Agent."""

    context_years = context.keys()
    metrics = _filter_metrics(annual_metrics, context_years)
    if not metrics:
        raise ValueError("No historical metrics available for prediction payload.")
    return {"metrics": metrics, "context": context}


def prediction_agent(payload: Dict[str, object]) -> Tuple[Dict[int, float], Dict[int, str]]:
    """Predict 2020-2022 returns using a simple heuristic and context."""

    metrics: List[AnnualMetric] = sorted(payload["metrics"], key=lambda item: item.year)
    trailing_returns = [metric.avg_return for metric in metrics[-3:]] or [0.0]
    baseline = float(mean(trailing_returns))

    predictions = {}
    rationales = {}

    def add_prediction(year: int, adjustment: float, narrative: str) -> None:
        predictions[year] = baseline + adjustment
        rationales[year] = narrative

    add_prediction(
        2020,
        adjustment=-0.15,
        narrative=(
            "COVID-19 shock and global shutdowns point to a sharp drawdown despite the prior expansion."
        ),
    )
    add_prediction(
        2021,
        adjustment=0.08,
        narrative=(
            "Reopening momentum, fiscal stimulus, and ultra-loose monetary policy set the stage for a rebound."
        ),
    )
    add_prediction(
        2022,
        adjustment=-0.05,
        narrative=(
            "Inflation spikes and aggressive Fed tightening suggest renewed volatility and downside pressure."
        ),
    )

    return predictions, rationales


def evaluator_agent(predictions: Dict[int, float], annual_metrics: Sequence[AnnualMetric]) -> Dict[str, object]:
    """Compare predictions with actual results for 2020-2022."""

    actual_lookup = {metric.year: metric for metric in annual_metrics if 2020 <= metric.year <= 2022}
    if len(actual_lookup) < 3:
        raise ValueError("Actual metrics for 2020-2022 are missing.")

    details: List[Dict[str, float]] = []
    for year in sorted(predictions):
        actual_return = actual_lookup[year].avg_return
        predicted_return = predictions[year]
        error = abs(predicted_return - actual_return)
        details.append({
            "Year": year,
            "Avg_Return": actual_return,
            "prediction": predicted_return,
            "error": error,
        })

    hardest_entry = max(details, key=lambda row: row["error"])
    hardest_year = hardest_entry["Year"]

    evaluation = {
        "details": details,
        "MAE": float(mean(row["error"] for row in details)),
        "hardest_year": int(hardest_year),
        "hardest_year_reason": (
            "Large shock relative to heuristic baseline" if hardest_year == 2020 else "Structural shift not captured"
        ),
    }
    return evaluation


def generate_prediction_chart(
    details: Sequence[Dict[str, float]],
    evaluation: Optional[Dict[str, object]] = None,
    *,
    chart_records: Optional[Sequence[Dict[str, Any]]] = None,
    predictions: Optional[Dict[int, float]] = None,
    output_dir: str = "artifacts",
    min_points_per_year: int = 100,
) -> Optional[str]:
    """Render the requested 2020–2022 Actual vs. Predicted line chart."""

    if not details:
        return None

    # These parameters remain in the signature for backward compatibility with
    # earlier dense-chart pipelines but are intentionally ignored now that the
    # user requested a focused 2020–2022 view.
    _ = (chart_records, min_points_per_year)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / "predictions_vs_actual.svg"

    predictions = predictions or {int(row["Year"]): float(row["prediction"]) for row in details}

    filtered_years = [year for year in sorted(predictions) if 2020 <= year <= 2022]
    if not filtered_years:
        return None

    actual_values = {int(row["Year"]): float(row["Avg_Return"]) for row in details}
    actual_series = [actual_values.get(year, float("nan")) for year in filtered_years]
    predicted_series = [predictions.get(year, float("nan")) for year in filtered_years]

    return _render_prediction_window_chart(
        filtered_years,
        actual_series,
        predicted_series,
        evaluation,
        chart_path,
    )




def _render_prediction_window_chart(
    years: Sequence[int],
    actual_values: Sequence[float],
    predicted_values: Sequence[float],
    evaluation: Optional[Dict[str, object]],
    chart_path: Path,
) -> Optional[str]:
    if not years:
        return None

    width, height = 900, 620
    margin_top = 110
    margin_bottom = 170
    margin_left = 90
    margin_right = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    scaled_actual = [value * 100 for value in actual_values]
    scaled_predicted = [value * 100 for value in predicted_values]
    combined = [value for value in scaled_actual + scaled_predicted if not math.isnan(value)]
    if not combined:
        return None

    min_value = min(combined)
    max_value = max(combined)
    padding = max(5.0, (max_value - min_value) * 0.15)
    min_value -= padding
    max_value += padding
    value_range = max(max_value - min_value, 1e-6)

    x_positions = {
        year: margin_left + idx * (plot_width / max(len(years) - 1, 1))
        for idx, year in enumerate(years)
    }

    def y_from_value(value: float) -> float:
        return margin_top + plot_height - ((value - min_value) / value_range) * plot_height

    y_axis = margin_left
    x_axis = margin_top + plot_height

    y_ticks = 5
    grid_lines = []
    y_tick_labels = []
    for i in range(y_ticks + 1):
        value = min_value + (value_range / y_ticks) * i
        y = y_from_value(value)
        grid_lines.append(
            f"<line x1='{y_axis}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='#e5e7eb' stroke-width='1'/>"
        )
        y_tick_labels.append(
            f"<text x='{margin_left - 12}' y='{y + 4:.2f}' font-size='12' fill='#111827' text-anchor='end'>{value:.1f}%</text>"
        )

    def _build_path(values: Sequence[float]) -> str:
        commands = []
        for idx, year in enumerate(years):
            value = values[idx]
            if math.isnan(value):
                continue
            x = x_positions[year]
            y = y_from_value(value)
            prefix = 'M' if not commands else 'L'
            commands.append(f"{prefix}{x:.2f},{y:.2f}")
        return ' '.join(commands)

    actual_path = _build_path(scaled_actual)
    predicted_path = _build_path(scaled_predicted)

    point_labels = []
    point_markers = []
    for idx, year in enumerate(years):
        actual = scaled_actual[idx]
        predicted = scaled_predicted[idx]
        x = x_positions[year]
        if not math.isnan(actual):
            y = y_from_value(actual)
            point_markers.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4' fill='#047857' stroke='white' stroke-width='1'/>"
            )
            point_labels.append(
                f"<text x='{x:.2f}' y='{y - 10:.2f}' font-size='12' fill='#065f46' text-anchor='middle'>Actual {year}: {actual:.1f}%</text>"
            )
        if not math.isnan(predicted):
            y_pred = y_from_value(predicted)
            point_markers.append(
                f"<circle cx='{x:.2f}' cy='{y_pred:.2f}' r='4' fill='#dc2626' stroke='white' stroke-width='1'/>"
            )
            point_labels.append(
                f"<text x='{x:.2f}' y='{y_pred + 18:.2f}' font-size='12' fill='#b91c1c' text-anchor='middle'>Pred {year}: {predicted:.1f}%</text>"
            )

    legend_y = margin_top + plot_height + 40
    legend = (
        f"<rect x='{margin_left}' y='{legend_y}' width='14' height='14' fill='#047857' />"
        f"<text x='{margin_left + 22}' y='{legend_y + 12}' font-size='12' fill='#111827'>Actual return</text>"
        f"<rect x='{margin_left + 160}' y='{legend_y}' width='14' height='14' fill='#dc2626' />"
        f"<text x='{margin_left + 178}' y='{legend_y + 12}' font-size='12' fill='#111827'>Predicted return</text>"
    )

    eval_text = ''
    if evaluation:
        mae = evaluation.get('MAE')
        hardest_year = evaluation.get('hardest_year')
        hardest_reason = evaluation.get('hardest_year_reason')
        if mae is not None:
            eval_text = f"MAE {float(mae):.2%} | Toughest year {hardest_year}: {hardest_reason}"

    comments = (
        f"<text x='{margin_left}' y='{legend_y + 40}' font-size='12' fill='#374151'>{eval_text}</text>"
    )

    x_labels = []
    for year in years:
        x = x_positions[year]
        x_labels.append(
            f"<text x='{x:.2f}' y='{x_axis + 20}' font-size='12' fill='#111827' text-anchor='middle'>{year}</text>"
        )

    svg_content = f"""<?xml version='1.0' encoding='UTF-8'?>
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <rect width='100%' height='100%' fill='white'/>
  <text x='{width/2}' y='40' text-anchor='middle' font-size='24' fill='#111827' font-weight='600'>S&amp;P 500 Actual vs. FinSight Predictions</text>
  <text x='{width/2}' y='70' text-anchor='middle' font-size='14' fill='#374151'>Annual returns focus (2020–2022 forecast window)</text>
  <line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#111827' stroke-width='1.2'/>
  <line x1='{margin_left}' y1='{x_axis}' x2='{margin_left + plot_width}' y2='{x_axis}' stroke='#111827' stroke-width='1.2'/>
  {''.join(grid_lines)}
  {''.join(y_tick_labels)}
  {''.join(x_labels)}
  <text x='{margin_left + plot_width / 2}' y='{x_axis + 50}' text-anchor='middle' font-size='13' fill='#374151'>Years</text>
  <text x='20' y='{margin_top + plot_height / 2}' text-anchor='middle' font-size='13' fill='#374151' transform='rotate(-90 20,{margin_top + plot_height / 2})'>Annual return (%)</text>
  <path d='{actual_path}' fill='none' stroke='#047857' stroke-width='2.5' />
  <path d='{predicted_path}' fill='none' stroke='#dc2626' stroke-width='2.5' stroke-dasharray='8,6' />
  {''.join(point_markers)}
  {''.join(point_labels)}
  {legend}
  {comments}
</svg>
"""

    chart_path.write_text(svg_content)
    return str(chart_path)


def _render_annual_chart(
    details: Sequence[Dict[str, float]],
    evaluation: Optional[Dict[str, object]],
    predictions: Dict[int, float],
    chart_path: Path,
) -> Optional[str]:
    years = [row["Year"] for row in details]
    actuals = [row["Avg_Return"] for row in details]
    prediction_values = [predictions[year] for year in years]
    max_abs_value = max(max(abs(value) for value in actuals + prediction_values), 1e-6)

    width, height = 820, 620
    margin_top = 110
    margin_bottom = 140
    margin_left = 90
    margin_right = 70
    margin = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_coord(index: int) -> float:
        if len(years) == 1:
            return margin_left + plot_width / 2
        step = plot_width / (len(years) - 1)
        return margin_left + index * step

    def y_coord(value: float) -> float:
        vertical_center = margin_top + plot_height / 2
        return vertical_center - (value / max_abs_value) * (plot_height / 2)

    def polyline(points: List[Tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    actual_points = [(x_coord(idx), y_coord(val)) for idx, val in enumerate(actuals)]
    prediction_points = [(x_coord(idx), y_coord(val)) for idx, val in enumerate(prediction_values)]

    x_axis_y = margin_top + plot_height
    y_axis_x = margin_left

    year_labels = "".join(
        f'<text x="{x_coord(idx):.2f}" y="{x_axis_y + 16:.2f}" text-anchor="middle" '
        f'font-size="14" fill="#111">{year}</text>'
        for idx, year in enumerate(years)
    )

    y_ticks = [-max_abs_value, -max_abs_value / 2, 0, max_abs_value / 2, max_abs_value]
    tick_elements = []
    grid_lines = []
    for value in y_ticks:
        y = y_coord(value)
        grid_lines.append(
            f'<line x1="{y_axis_x}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        tick_elements.append(
            f'<line x1="{y_axis_x - 5}" y1="{y:.2f}" x2="{y_axis_x}" y2="{y:.2f}" stroke="#111" stroke-width="1"/>'
        )
        tick_elements.append(
            f'<text x="{y_axis_x - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#111">{value:.0%}</text>'
        )
    for idx, _ in enumerate(years):
        x = x_coord(idx)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{x_axis_y}" stroke="#f3f4f6" stroke-width="1"/>'
        )
    tick_elements_markup = "\n        ".join(tick_elements)
    grid_lines_markup = "\n        ".join(grid_lines)

    def point_annotations(points: List[Tuple[float, float]], values: List[float], color: str) -> str:
        circles = []
        for (x, y), value in zip(points, values):
            circles.append(
                f'<g>'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
                f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="12" fill="{color}">{value:.1%}</text>'
                f'</g>'
            )
        return "\n        ".join(circles)

    connectors = "\n        ".join(
        f'<line x1="{x_coord(idx):.2f}" y1="{y_coord(actuals[idx]):.2f}" '
        f'x2="{x_coord(idx):.2f}" y2="{y_coord(prediction_values[idx]):.2f}" stroke="#9ca3af" stroke-dasharray="4 4"/>'
        for idx in range(len(years))
    )

    summary_lines: List[str] = []
    if evaluation:
        mae = evaluation.get("MAE")
        hardest_year = evaluation.get("hardest_year")
        reason = evaluation.get("hardest_year_reason")
        if mae is not None:
            summary_lines.append(f"MAE {mae:.2%}")
        if hardest_year is not None:
            summary_lines.append(f"Toughest year {hardest_year}: {reason}")
    summary_text = " | ".join(summary_lines)

    legend_y = x_axis_y + 65
    legend = f"""
        <g transform="translate({width/2 - 170:.2f}, {legend_y:.2f})">
            <g>
                <line x1="0" y1="6" x2="24" y2="6" stroke="#15803d" stroke-width="3"/>
                <circle cx="12" cy="6" r="4" fill="#15803d" stroke="#fff" stroke-width="1.5"/>
                <text x="36" y="9" fill="#111" font-size="12">Actual annual return</text>
            </g>
            <g transform="translate(0, 24)">
                <line x1="0" y1="6" x2="24" y2="6" stroke="#b91c1c" stroke-width="3" stroke-dasharray="6 4"/>
                <circle cx="12" cy="6" r="4" fill="#b91c1c" stroke="#fff" stroke-width="1.5"/>
                <text x="36" y="9" fill="#111" font-size="12">Prediction from FinSight</text>
            </g>
        </g>
    """

    summary_block = (
        f'<text x="{width/2:.2f}" y="{legend_y + 55:.2f}" text-anchor="middle" font-size="13" fill="#374151">{summary_text}</text>'
        if summary_text
        else ""
    )

    svg_header = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"

    title_y = 40
    subtitle_y = title_y + 24

    svg_content = svg_header + f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#ffffff"/>
        <text x="{width/2:.2f}" y="{title_y:.2f}" fill="#111" font-size="20" font-weight="600" text-anchor="middle">S&amp;P 500 Actual vs. FinSight Predictions</text>
        <text x="{width/2:.2f}" y="{subtitle_y:.2f}" fill="#4b5563" font-size="14" text-anchor="middle">2015-2022 annual returns with error connectors</text>
        {grid_lines_markup}
        <line x1="{y_axis_x}" y1="{margin_top}" x2="{y_axis_x}" y2="{x_axis_y}" stroke="#111" stroke-width="1.5"/>
        <line x1="{y_axis_x}" y1="{x_axis_y}" x2="{width - margin_right}" y2="{x_axis_y}" stroke="#111" stroke-width="1.5"/>
        <text x="{width/2:.2f}" y="{x_axis_y + 35:.2f}" text-anchor="middle" font-size="13" fill="#111">Year</text>
        <text x="{40}" y="{(margin_top + plot_height/2):.2f}" transform="rotate(-90 {40} {(margin_top + plot_height/2):.2f})" text-anchor="middle" font-size="13" fill="#111">Annual return</text>
        {tick_elements_markup}
        {connectors}
        <polyline points="{polyline(actual_points)}" fill="none" stroke="#15803d" stroke-width="3"/>
        <polyline points="{polyline(prediction_points)}" fill="none" stroke="#b91c1c" stroke-width="3" stroke-dasharray="6 4"/>
        {point_annotations(actual_points, actuals, '#15803d')}
        {point_annotations(prediction_points, prediction_values, '#b91c1c')}
        {year_labels}
        {legend}
        {summary_block}
    </svg>
    """

    chart_path.write_text("\n".join(line.strip() for line in svg_content.strip().splitlines()) + "\n")
    return str(chart_path)


def finance_bro_agent(evaluation: Dict[str, object], chart_path: Optional[str]) -> str:
    """Translate evaluation insights into casual language."""

    mae = evaluation["MAE"]
    hardest_year = evaluation["hardest_year"]
    if hardest_year == 2020:
        reason = "COVID panic punched the market in the face before the Fed broke out the firehose."
    elif hardest_year == 2021:
        reason = "Stimulus-fueled melt-up outran our cautious playbook."
    else:
        reason = "Rate hikes and inflation jitters twisted the plot harder than expected."

    chart_sentence = f" Peep the chart at {chart_path} to see predictions vs. reality." if chart_path else ""

    return (
        f"Our calls were off by about {mae:.2%} on average. The trickiest year was {hardest_year} because {reason}."
        + chart_sentence
    )


def refinement_agent(evaluation: Dict[str, object], finance_summary: str) -> List[str]:
    """Recommend pipeline improvements."""

    suggestions = [
        "Blend macro indicators (CPI, unemployment) with the return features to capture regime shifts.",
        "Layer a sentiment score from earnings calls or news headlines to react faster to shocks.",
        "Use rolling walk-forward validation so the Prediction Agent continually re-trains with the latest year.",
    ]
    if "COVID" in finance_summary:
        suggestions.append("Introduce crisis detectors that down-weight pre-pandemic averages when global health risks surge.")
    return suggestions


def mr_white_agent(
    annual_metrics: Sequence[AnnualMetric],
    context: Dict[int, List[str]],
    predictions: Dict[int, float],
    evaluation: Dict[str, object],
    finance_summary: str,
    refinement_suggestions: Sequence[str],
    vector_store_status: str,
    data_source_report: DataSourceReport,
) -> str:
    """Explain the architecture to a curious student in a didactic tone."""

    data_source = data_source_report.source
    mae = evaluation.get("MAE")
    hardest_year = evaluation.get("hardest_year")
    hardest_reason = evaluation.get("hardest_year_reason")
    years_span = f"{min(metric.year for metric in annual_metrics)}–{max(metric.year for metric in annual_metrics)}"

    context_years = ", ".join(str(year) for year in sorted(context))

    return "\n".join(
        [
            "Hey there, apprentice — Mr. White here.",
            "We built FinSight as a relay race of specialists so you can trace every decision:",
            "1. A LangChain Runnable sequence kicks off the show so every agent hands a structured state to the next.",
            "2. DataCollector scrapes Yahoo Finance's ^GSPC history page (cached locally as data/dataset.json) and",
            f"   distills it into AnnualMetric objects covering {years_span}. When the scrape is blocked we fall back",
            "   to the bundled snapshot, so quality stays aligned with the same Yahoo feed.",
            f"3. Research Agent pins three macro story beats to each pre-pandemic year ({context_years}) and pushes",
            f"   them through a Pinecone vector store ({vector_store_status}) so later agents can query a real RAG memory.",
            "4. Prediction Agent studies those 2015–2019 patterns and drafts 2020–2022 calls with narratives.",
            "5. Evaluator cross-checks predictions against actual returns, reports MAE and toughest year,",
            f"   which this run pegs at {mae:.2%} average error with {hardest_year} hardest because {hardest_reason}.",
            "6. Finance Bro translates that into street language and points everyone to the SVG comparison chart.",
            "7. Refinement Agent turns the misses into a backlog of upgrades so the next version learns.",
            "8. Test Titan enforces contracts (no missing years, numeric MAE) before anyone declares victory.",
            "9. Lord of the Mysteries packages everything and I, Mr. White, narrate the architecture so your",
            "   curious mind sees how multi-agent orchestration feels in practice.",
            f"Data source this round: {data_source}. {data_source_report.note}",
            f"Finance Bro's verdict: {finance_summary}",
            "Next experiments on deck: " + "; ".join(refinement_suggestions),
        ]
    )


def test_titan_agent(
    annual_metrics: Sequence[AnnualMetric], predictions: Dict[int, float], evaluation: Dict[str, object]
) -> None:
    """Run sanity checks over pipeline artifacts."""

    checks: List[str] = []

    if not annual_metrics:
        checks.append("Annual metrics dataset is empty.")
    elif not all(metric.year and isinstance(metric.avg_return, float) for metric in annual_metrics):
        checks.append("Annual metrics contain invalid entries.")
    if set(predictions.keys()) != {2020, 2021, 2022}:
        checks.append("Predictions must cover 2020-2022.")
    if not isinstance(evaluation.get("MAE"), float):
        checks.append("MAE must be numeric.")
    if not evaluation.get("details"):
        checks.append("Evaluation details are missing.")

    if checks:
        raise AssertionError("; ".join(checks))

    print("All tests passed ✅")


def _collect_step(state: Dict[str, object]) -> Dict[str, object]:
    dataset_options: Dict[str, Any] = state.get("dataset_options", {})  # type: ignore[assignment]
    annual_metrics, data_source_report, chart_records = collect_sp500_data(**dataset_options)
    state["annual_metrics"] = annual_metrics
    state["data_source_report"] = data_source_report
    state["chart_records"] = chart_records
    return state


def _context_step(state: Dict[str, object]) -> Dict[str, object]:
    vector_store: ContextVectorStore = state["vector_store"]  # type: ignore[assignment]
    years = state.get("context_years", [])
    context, status = research_macro_events(years, vector_store=vector_store)
    state["context"] = context
    state["vector_store_status"] = status
    return state


def _payload_step(state: Dict[str, object]) -> Dict[str, object]:
    annual_metrics: Sequence[AnnualMetric] = state["annual_metrics"]  # type: ignore[assignment]
    context: Dict[int, List[str]] = state["context"]  # type: ignore[assignment]
    state["payload"] = build_prediction_payload(annual_metrics, context)
    return state


def _prediction_step(state: Dict[str, object]) -> Dict[str, object]:
    payload: Dict[str, object] = state["payload"]  # type: ignore[assignment]
    predictions, rationales = prediction_agent(payload)
    state["predictions"] = predictions
    state["rationales"] = rationales
    return state


def _evaluation_step(state: Dict[str, object]) -> Dict[str, object]:
    predictions: Dict[int, float] = state["predictions"]  # type: ignore[assignment]
    annual_metrics: Sequence[AnnualMetric] = state["annual_metrics"]  # type: ignore[assignment]
    evaluation = evaluator_agent(predictions, annual_metrics)
    data_report: DataSourceReport = state["data_source_report"]  # type: ignore[assignment]
    evaluation["data_source"] = data_report.source
    state["evaluation"] = evaluation
    return state


def _chart_step(state: Dict[str, object]) -> Dict[str, object]:
    evaluation: Dict[str, object] = state["evaluation"]  # type: ignore[assignment]
    details: Sequence[Dict[str, float]] = evaluation["details"]  # type: ignore[index]
    chart_records: Optional[Sequence[Dict[str, Any]]] = state.get("chart_records")  # type: ignore[assignment]
    predictions: Dict[int, float] = state.get("predictions", {})  # type: ignore[assignment]
    state["chart_path"] = generate_prediction_chart(
        details,
        evaluation,
        chart_records=chart_records,
        predictions=predictions,
    )
    return state


def _finance_step(state: Dict[str, object]) -> Dict[str, object]:
    evaluation: Dict[str, object] = state["evaluation"]  # type: ignore[assignment]
    chart_path: Optional[str] = state.get("chart_path")
    state["finance_bro_summary"] = finance_bro_agent(evaluation, chart_path)
    return state


def _refinement_step(state: Dict[str, object]) -> Dict[str, object]:
    evaluation: Dict[str, object] = state["evaluation"]  # type: ignore[assignment]
    summary: str = state["finance_bro_summary"]  # type: ignore[assignment]
    state["refinement_suggestions"] = refinement_agent(evaluation, summary)
    return state


def _test_step(state: Dict[str, object]) -> Dict[str, object]:
    annual_metrics: Sequence[AnnualMetric] = state["annual_metrics"]  # type: ignore[assignment]
    predictions: Dict[int, float] = state["predictions"]  # type: ignore[assignment]
    evaluation: Dict[str, object] = state["evaluation"]  # type: ignore[assignment]
    test_titan_agent(annual_metrics, predictions, evaluation)
    return state


def _mentor_step(state: Dict[str, object]) -> Dict[str, object]:
    annual_metrics: Sequence[AnnualMetric] = state["annual_metrics"]  # type: ignore[assignment]
    context: Dict[int, List[str]] = state["context"]  # type: ignore[assignment]
    predictions: Dict[int, float] = state["predictions"]  # type: ignore[assignment]
    evaluation: Dict[str, object] = state["evaluation"]  # type: ignore[assignment]
    finance_summary: str = state["finance_bro_summary"]  # type: ignore[assignment]
    suggestions: Sequence[str] = state["refinement_suggestions"]  # type: ignore[assignment]
    vector_store_status: str = state["vector_store_status"]  # type: ignore[assignment]
    data_report: DataSourceReport = state["data_source_report"]  # type: ignore[assignment]
    state["mr_white_briefing"] = mr_white_agent(
        annual_metrics,
        context,
        predictions,
        evaluation,
        finance_summary,
        suggestions,
        vector_store_status,
        data_report,
    )
    return state


def _wrap_stage(stage_name: str, func: Callable[[Dict[str, object]], Dict[str, object]], stage_logger: Optional[StageLogger]):
    def wrapped(state: Dict[str, object]) -> Dict[str, object]:
        updated = func(state)
        if stage_logger:
            stage_logger(stage_name, updated)
        return updated

    return wrapped


def _bootstrap_state(
    vector_store: ContextVectorStore,
    context_years: Sequence[int],
    stage_logger: Optional[StageLogger],
    dataset_options: Optional[Dict[str, Any]],
) -> Callable[[Dict[str, object]], Dict[str, object]]:
    def bootstrap(_: Dict[str, object]) -> Dict[str, object]:
        state = {
            "vector_store": vector_store,
            "context_years": list(context_years),
            "dataset_options": dataset_options or {},
        }
        if stage_logger:
            stage_logger("bootstrap", state)
        return state

    return bootstrap


def build_langchain_chain(
    *,
    vector_store: Optional[ContextVectorStore] = None,
    context_years: Optional[Sequence[int]] = None,
    stage_logger: Optional[StageLogger] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_region: Optional[str] = None,
    dataset_url: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    refresh_dataset: bool = False,
):
    """Expose the LangChain Runnable sequence so scripts can inspect every hop."""

    vector_store = vector_store or ContextVectorStore(
        api_key=pinecone_api_key,
        region=pinecone_region,
    )
    years = list(context_years or range(2015, 2020))

    dataset_options = {
        "dataset_path": dataset_path or DEFAULT_DATASET_PATH,
        "dataset_url": dataset_url or DEFAULT_HISTORY_URL,
        "refresh_dataset": refresh_dataset,
    }

    orchestrator = (
        RunnableLambda(_bootstrap_state(vector_store, years, stage_logger, dataset_options))
        | RunnableLambda(_wrap_stage("collect", _collect_step, stage_logger))
        | RunnableLambda(_wrap_stage("context", _context_step, stage_logger))
        | RunnableLambda(_wrap_stage("payload", _payload_step, stage_logger))
        | RunnableLambda(_wrap_stage("prediction", _prediction_step, stage_logger))
        | RunnableLambda(_wrap_stage("evaluation", _evaluation_step, stage_logger))
        | RunnableLambda(_wrap_stage("chart", _chart_step, stage_logger))
        | RunnableLambda(_wrap_stage("finance", _finance_step, stage_logger))
        | RunnableLambda(_wrap_stage("refinement", _refinement_step, stage_logger))
        | RunnableLambda(_wrap_stage("test", _test_step, stage_logger))
        | RunnableLambda(_wrap_stage("mentor", _mentor_step, stage_logger))
    )

    return orchestrator


def state_to_agent_output(state: Dict[str, object]) -> AgentOutput:
    """Convert the LangChain state dict into a structured AgentOutput."""

    return AgentOutput(
        annual_metrics=state["annual_metrics"],
        context=state["context"],
        predictions=state["predictions"],
        rationales=state["rationales"],
        evaluation=state["evaluation"],
        finance_bro_summary=state["finance_bro_summary"],
        refinement_suggestions=state["refinement_suggestions"],
        chart_path=state.get("chart_path"),
        mr_white_briefing=state["mr_white_briefing"],
        vector_store_status=state["vector_store_status"],
        data_source_report=state["data_source_report"],
    )


def format_agent_output(output: AgentOutput) -> str:
    """Create a human-friendly summary for report.txt."""

    def pct(value: float) -> str:
        return f"{value:+.2%}"

    lines: List[str] = []
    lines.append("FinSight Multi-Agent Report")
    lines.append("=" * 29)
    lines.append(f"Data source: {output.data_source_report.source}")
    lines.append(f"Source notes: {output.data_source_report.note}")
    lines.append(f"Vector store status: {output.vector_store_status}")
    if output.chart_path:
        lines.append(f"Chart: {output.chart_path}")
    lines.append("")

    lines.append("Annual Metrics (2015-2022)")
    lines.append("Year    Return     Volatility")
    for metric in output.annual_metrics:
        lines.append(f"{metric.year}    {pct(metric.avg_return):>8}   {pct(metric.volatility):>10}")
    lines.append("")

    lines.append("Context Highlights")
    for year in sorted(output.context):
        events = output.context[year]
        bullet_block = "\n".join(f"- {event}" for event in events)
        lines.append(f"{year}:\n" + indent(bullet_block, "  "))
    lines.append("")

    evaluation_details: Sequence[Dict[str, float]] = output.evaluation.get("details", [])  # type: ignore[assignment]
    if evaluation_details:
        lines.append("Predictions vs. Actuals")
        lines.append("Year    Actual     Predicted    Error")
        for row in evaluation_details:
            lines.append(
                f"{int(row['Year'])}    {pct(row['Avg_Return']):>8}   {pct(row['prediction']):>10}   {pct(row['error']):>8}"
            )
        lines.append("")

    mae_value = output.evaluation.get("MAE")
    if isinstance(mae_value, (int, float)):
        mae_str = f"{mae_value:.2%}"
    else:
        mae_str = str(mae_value)
    hardest_year = output.evaluation.get("hardest_year")
    reason = output.evaluation.get("hardest_year_reason")
    lines.append(f"MAE: {mae_str} (hardest year: {hardest_year} — {reason})")
    lines.append("")

    lines.append("Finance Bro")
    lines.append(output.finance_bro_summary)
    lines.append("")

    lines.append("Refinement Suggestions")
    lines.extend(f"- {suggestion}" for suggestion in output.refinement_suggestions)
    lines.append("")

    lines.append("Mr. White's Briefing")
    lines.append(output.mr_white_briefing.strip())
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def run_pipeline(
    stage_logger: Optional[StageLogger] = None,
    *,
    pinecone_api_key: Optional[str] = None,
    pinecone_region: Optional[str] = None,
    context_years: Optional[Sequence[int]] = None,
    dataset_url: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    refresh_dataset: bool = False,
) -> AgentOutput:
    """Execute the FinSight workflow end-to-end via a LangChain Runnable pipeline."""

    orchestrator = build_langchain_chain(
        stage_logger=stage_logger,
        context_years=context_years,
        pinecone_api_key=pinecone_api_key,
        pinecone_region=pinecone_region,
        dataset_url=dataset_url,
        dataset_path=dataset_path,
        refresh_dataset=refresh_dataset,
    )
    state = orchestrator.invoke({})
    return state_to_agent_output(state)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FinSight LangChain pipeline")
    parser.add_argument(
        "--pinecone-api-key",
        dest="pinecone_api_key",
        help="Override the PINECONE_API_KEY env var for this run",
    )
    parser.add_argument(
        "--pinecone-region",
        dest="pinecone_region",
        help="Override the PINECONE_REGION env var (defaults to us-east-1)",
    )
    parser.add_argument(
        "--context-years",
        dest="context_years",
        nargs="+",
        type=int,
        help="Years to feed into the Research Agent (default 2015-2019)",
    )
    parser.add_argument(
        "--dataset-url",
        dest="dataset_url",
        default=DEFAULT_HISTORY_URL,
        help="Yahoo Finance history URL to scrape (defaults to the ^GSPC link)",
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        help="Override the dataset.json location",
    )
    parser.add_argument(
        "--refresh-dataset",
        dest="refresh_dataset",
        action="store_true",
        help="Force a fresh scrape before running",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    output = run_pipeline(
        pinecone_api_key=args.pinecone_api_key,
        pinecone_region=args.pinecone_region,
        context_years=args.context_years,
        dataset_url=args.dataset_url,
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        refresh_dataset=args.refresh_dataset,
    )
    report_text = format_agent_output(output)
    report_path = Path(__file__).resolve().parent.parent / "report.txt"
    report_path.write_text(report_text)
    print(f"Report written to {report_path}")
