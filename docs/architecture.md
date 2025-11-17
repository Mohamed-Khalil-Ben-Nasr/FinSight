# FinSight System Architecture

## Pipeline Overview
FinSight orchestrates a sequence of specialized agents to transform raw S&P 500 market data into predictions, evaluations, and end-user friendly explanations. The workflow is intentionally modular so every agent has a narrow, testable contract.

1. **Data Acquisition & Preparation**
   - **Agent:** DataCollector
   - **Input:** Ticker symbol (^GSPC), date range (2015-01-01 to 2022-12-31).
   - **Process:** Download daily OHLC data via `yfinance`, compute daily returns, aggregate into annual return and volatility metrics, and log the data source (`yfinance` vs. cached snapshot). The fallback JSON was generated from the same Yahoo Finance series, so quality remains tied to a reputable public feed while ensuring offline reproducibility.
   - **Output:** List of `AnnualMetric(year, avg_return, volatility)` objects spanning 2015–2022, plus a label describing whether the data was live or cached.

2. **Context Enrichment**
   - **Agent:** Research Agent
   - **Input:** Years 2015–2019.
   - **Process:** Retrieve or synthesize the three most impactful macro events for each year, embed them using a deterministic hash, and upsert them into the Pinecone `finsight-market-events` serverless index (or the offline in-memory replica when credentials are absent). The summaries are fetched right back out so downstream agents always consume context through the vector-store interface.
   - **Output:** Dictionary keyed by year containing bullet-point summaries + Pinecone status metadata.

3. **Prediction Preparation**
   - **Agent:** LangChain Orchestrator (a RunnableLambda step implemented by the Coding Demon)
   - **Input:** Historical annual metrics (2015–2019) and the Pinecone-backed context dictionary.
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
