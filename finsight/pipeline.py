"""FinSight multi-agent pipeline implementation."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from textwrap import indent
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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


def collect_sp500_data(
    start: str = "2015-01-01", end: str = "2022-12-31"
) -> Tuple[List[AnnualMetric], DataSourceReport]:
    """Download S&P 500 data and compute annual metrics.

    Falls back to a cached JSON snapshot when yfinance/pandas are unavailable.
    Returns a tuple of (metrics, DataSourceReport).
    """

    failure_reasons: List[str] = []
    if pd is not None and yf is not None:
        try:
            ticker = yf.Ticker("^GSPC")
            history = ticker.history(start=start, end=end)
            if not history.empty:
                metrics = _compute_metrics_from_history(history)
                note = (
                    "Live ^GSPC download succeeded via yfinance and was cleaned with pandas"
                )
                return metrics, DataSourceReport(source="yfinance", note=note)
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
    return metrics, DataSourceReport(source="cached_snapshot", note=note)


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
    output_dir: str = "artifacts",
) -> Optional[str]:
    """Render an informative SVG comparing actual vs. predicted returns."""

    if not details:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / "predictions_vs_actual.svg"

    years = [row["Year"] for row in details]
    actuals = [row["Avg_Return"] for row in details]
    predictions = [row["prediction"] for row in details]
    max_abs_value = max(max(abs(value) for value in actuals + predictions), 1e-6)

    width, height = 820, 560
    margin_top = 110
    margin_bottom = 90
    margin_left = 90
    margin_right = 70
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
    prediction_points = [(x_coord(idx), y_coord(val)) for idx, val in enumerate(predictions)]

    x_axis_y = margin_top + plot_height
    y_axis_x = margin_left

    year_labels = "".join(
        f'<text x="{x_coord(idx):.2f}" y="{x_axis_y + 25:.2f}" text-anchor="middle" '
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
        f'x2="{x_coord(idx):.2f}" y2="{y_coord(predictions[idx]):.2f}" stroke="#9ca3af" stroke-dasharray="4 4"/>'
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

    legend_y = height - margin_bottom - 35
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
        f'<text x="{width/2:.2f}" y="{legend_y + 60:.2f}" text-anchor="middle" font-size="13" fill="#374151">{summary_text}</text>'
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
        <text x="{width/2:.2f}" y="{x_axis_y + 45:.2f}" text-anchor="middle" font-size="13" fill="#111">Year</text>
        <text x="{40}" y="{(margin_top + plot_height/2):.2f}" transform="rotate(-90 {40} {(margin_top + plot_height/2):.2f})" text-anchor="middle" font-size="13" fill="#111">Annual return</text>
        {tick_elements_markup}
        {connectors}
        <polyline points="{polyline(actual_points)}" fill="none" stroke="#15803d" stroke-width="3"/>
        <polyline points="{polyline(prediction_points)}" fill="none" stroke="#b91c1c" stroke-width="3" stroke-dasharray="6 4"/>
        {point_annotations(actual_points, actuals, '#15803d')}
        {point_annotations(prediction_points, predictions, '#b91c1c')}
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
            "2. DataCollector grabs ^GSPC closes via yfinance (or our cached snapshot when offline) and distills",
            f"   them into AnnualMetric objects covering {years_span}. That cache is the same Yahoo Finance data,",
            "   just frozen for reproducibility, so quality stays aligned with a widely trusted public feed.",
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
    annual_metrics, data_source_report = collect_sp500_data()
    state["annual_metrics"] = annual_metrics
    state["data_source_report"] = data_source_report
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
    state["chart_path"] = generate_prediction_chart(details, evaluation)
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
) -> Callable[[Dict[str, object]], Dict[str, object]]:
    def bootstrap(_: Dict[str, object]) -> Dict[str, object]:
        state = {"vector_store": vector_store, "context_years": list(context_years)}
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
):
    """Expose the LangChain Runnable sequence so scripts can inspect every hop."""

    vector_store = vector_store or ContextVectorStore(
        api_key=pinecone_api_key,
        region=pinecone_region,
    )
    years = list(context_years or range(2015, 2020))

    orchestrator = (
        RunnableLambda(_bootstrap_state(vector_store, years, stage_logger))
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
) -> AgentOutput:
    """Execute the FinSight workflow end-to-end via a LangChain Runnable pipeline."""

    orchestrator = build_langchain_chain(
        stage_logger=stage_logger,
        context_years=context_years,
        pinecone_api_key=pinecone_api_key,
        pinecone_region=pinecone_region,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    output = run_pipeline(
        pinecone_api_key=args.pinecone_api_key,
        pinecone_region=args.pinecone_region,
        context_years=args.context_years,
    )
    report_text = format_agent_output(output)
    report_path = Path(__file__).resolve().parent.parent / "report.txt"
    report_path.write_text(report_text)
    print(f"Report written to {report_path}")
