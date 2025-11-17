# FinSight System Architecture

## Pipeline Overview
FinSight orchestrates a sequence of specialized agents to transform raw S&P 500 market data into predictions, evaluations, and end-user friendly explanations. The workflow is intentionally modular so every agent has a narrow, testable contract.

1. **Data Acquisition & Preparation**
   - **Agent:** DataCollector
   - **Input:** Ticker symbol (^GSPC), date range (2015-01-01 to 2022-12-31).
   - **Process:** Download daily OHLC data via `yfinance`, compute daily returns, aggregate into annual return and volatility metrics. When network access or heavy dependencies are unavailable, fall back to a cached JSON snapshot so the rest of the system can still run deterministically.
   - **Output:** List of `AnnualMetric(year, avg_return, volatility)` objects spanning 2015–2022, plus a label describing whether the data was live or cached.

2. **Context Enrichment**  
   - **Agent:** Research Agent  
   - **Input:** Years 2015–2019.  
   - **Process:** Retrieve or synthesize the three most impactful macro events for each year. Persist summaries into the `market-events` vector store (Pinecone) when available.  
   - **Output:** Dictionary keyed by year containing bullet-point summaries.

3. **Prediction Preparation**  
   - **Agent:** Solutions Architect + Coding Demon  
   - **Input:** Historical annual metrics (2015–2019) and macro summaries.  
   - **Process:** Align numeric features with qualitative context to form a feature pack for downstream reasoning agents.  
   - **Output:** Structured payload `{"metrics": df_2015_2019, "context": events}`.

4. **Forecasting**  
   - **Agent:** Prediction Agent  
   - **Input:** Feature pack from Step 3.  
   - **Process:** Apply reasoning/heuristics over historical patterns and macro context to infer annual returns for 2020–2022.  
   - **Output:** Dict `{2020: pred_return, 2021: ..., 2022: ...}` with textual rationale per year.

5. **Evaluation**  
   - **Agent:** Evaluator  
   - **Input:** Predictions from Step 4 and actual metrics from Step 1.  
   - **Process:** Compute Mean Absolute Error (MAE), highlight most difficult year and diagnostic comments.  
   - **Output:** Evaluation report containing MAE, per-year error, and insights.

6. **Communication & Visualization**
   - **Agent:** Finance Bro (paired with a lightweight SVG chart generator)
   - **Input:** Evaluation report (including per-year actual vs. predicted returns).
   - **Process:** Render a quick comparison chart (SVG stored under `artifacts/`) and translate technical findings into plain-English summary that references the visualization.
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

## Constraints & Checks
- Every agent’s function should raise informative errors when contract assumptions are violated (e.g., missing columns, empty predictions).
- The pipeline should be runnable end-to-end via a single `python -m finsight.pipeline` command for reproducibility.
- Testing hooks must be easily accessible so Test Titan can run sanity checks without network calls (mock/stub where necessary).
