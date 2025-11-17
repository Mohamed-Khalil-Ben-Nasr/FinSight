# FinSight System Architecture

## Pipeline Overview
FinSight orchestrates a sequence of specialized agents to transform raw S&P 500 market data into predictions, evaluations, and end-user friendly explanations. The workflow is intentionally modular so every agent has a narrow, testable contract.

1. **Data Acquisition & Preparation**
   - **Agent:** DataCollector
   - **Input:** Yahoo Finance historical page for ^GSPC (period1=1420070400, period2=1668643200) + date range filter (2015-01-01 to 2022-12-31).
   - **Process:** Scrape the page’s `HistoricalPriceStore` JSON into `data/dataset.json`, compute daily returns from the cached candles, aggregate into annual return/volatility metrics, and log whether the run used the scraped dataset (`yahoo_history_html`), a live `yfinance` pull, or the legacy fallback snapshot.
   - **Output:** List of `AnnualMetric(year, avg_return, volatility)` objects spanning 2015–2022, plus a `DataSourceReport` describing the provenance.

2. **Context Enrichment**
   - **Agent:** Research Agent
   - **Input:** Years 2015–2019.
   - **Process:** Retrieve or synthesize the three most impactful macro events for each year, embed them using a deterministic hash, and upsert them into the Pinecone `finsight-market-events` serverless index (or the offline in-memory replica when credentials are absent). The summaries are fetched right back out so downstream agents always consume context through the vector-store interface.
   - **Output:** Dictionary keyed by year containing bullet-point summaries + Pinecone status metadata.

3. **Prediction Preparation**
   - **Agent:** LangChain Orchestrator (a RunnableLambda step implemented by the Coding Demon)
   - **Input:** Historical annual metrics (2015–2019), Pinecone-backed context, and the raw daily candles scraped in Step 1.
   - **Process:** Align numeric features with qualitative context, then bundle the per-day candles so the Prediction Agent can operate at trading-day granularity.
   - **Output:** Structured payload `{"metrics": df_2015_2019, "context": events, "chart_records": candles}`.

4. **Forecasting**
   - **Agent:** Prediction Agent
   - **Input:** Feature pack from Step 3.
   - **Process:** Apply reasoning/heuristics over historical patterns and macro context to infer **daily** price paths for every trading session in 2020–2022 (capped/adjusted per year), then derive the annual returns. Persist the trading-day predictions to `artifacts/daily_predictions_2020_2022.json` for inspection.
   - **Output:** Dict `{2020: pred_return, 2021: ..., 2022: ...}` with textual rationale per year **plus** the saved daily predictions dataset.

5. **Evaluation**
   - **Agent:** Evaluator
   - **Input:** Predictions from Step 4, actual metrics from Step 1, and the aligned daily candles.
   - **Process:** Compute Mean Absolute Error (MAE), daily Mean Absolute Percentage Error (MAPE) across every overlapping trading day, and highlight the most difficult year with diagnostic comments.
   - **Output:** Evaluation report containing MAE, daily MAPE, per-year error, and insights.

6. **Communication & Visualization**
   - **Agent:** Finance Bro (paired with a lightweight SVG chart generator)
   - **Input:** Evaluation report (including per-year and daily stats) plus the daily prediction dataset.
   - **Process:** Render a daily Actual vs. Predicted comparison chart (SVG stored under `artifacts/`) and translate technical findings—including daily MAPE—into a plain-English summary that references the visualization.
   - **Output:** Friendly narrative for non-technical audiences plus an embeddable chart path.

7. **Testing & Validation**  
   - **Agent:** Test Titan  
   - **Input:** All intermediate artifacts.  
   - **Process:** Validate schema conformity, numeric sanity (no NaNs, MAE numeric, predictions cover requested years).  
   - **Output:** Pass/fail checklist.

8. **Refinement**
   - **Agent:** Refinement Agent
   - **Input:** Evaluator + Finance Bro outputs.
   - **Process:** Recommend improvements—additional indicators, alternative training windows, or integration tweaks.
   - **Output:** Actionable refinement backlog.

9. **Mentorship & Narrative Packaging**
   - **Agent:** Mr White
   - **Input:** Everything above (metrics, context, predictions, evaluation, Finance Bro notes, refinement list).
   - **Process:** Summarize the architecture for a curious student so newcomers can see how the multi-agent relay works end-to-end.
   - **Output:** Didactic walkthrough embedded in the final `AgentOutput`.

10. **LangChain Orchestration Layer**
    - **Agent:** Lord of the Mysteries (implemented via LangChain `RunnableLambda` sequence)
    - **Input:** Empty dict (LangChain injects the working state).
    - **Process:** Chains `_collect_step → _context_step → … → _mentor_step`, ensuring each agent receives the structured state it expects without manual glue code.
    - **Output:** Final state dictionary that `AgentOutput` consumes. The helper `build_langchain_chain()` exposes this Runnable so scripts (see `scripts/run_langchain_workflow.py`) can attach a `stage_logger` and print the exact baton pass at every hop.

## Constraints & Checks
- Every agent’s function should raise informative errors when contract assumptions are violated (e.g., missing columns, empty predictions).
- The pipeline should be runnable end-to-end via a single `python -m finsight.pipeline` command for reproducibility.
- Testing hooks must be easily accessible so Test Titan can run sanity checks without network calls (mock/stub where necessary).
- Pinecone usage must fail gracefully (fall back to in-memory) while still reporting the vector-store status to the mentorship and README logs so interviewers can verify LangChain + Pinecone participation.
- CLI overrides (`--pinecone-api-key`, `--pinecone-region`) allow demonstrations to temporarily inject credentials without touching shell profiles, which is especially useful during interviews on managed laptops.
