# FinSight

FinSight is a multi-agent AI workflow that autonomously collects, analyzes, and forecasts S&P 500 performance. It was designed as an interview-ready showcase, so this README doubles as the system’s report: you will find the project proposal, the agent roster, the workflow explanation, and the instructions for running everything locally—including how to view the predictions-versus-actuals chart the Finance Bro references.

## Project Proposal
- **Objective:** Blend autonomous agents, quantitative data, and contextual reasoning to forecast the S&P 500’s annual returns, using 2015–2019 data to predict 2020–2022.
- **Motivation:** Demonstrate how modern AI infrastructure can support financial intelligence pipelines, prediction markets, or research workflows that require explainable insights.
- **Deliverables:** A reproducible pipeline that outputs (1) the full agent conversation, (2) an evaluation with MAE/error diagnostics, (3) a Finance Bro summary, (4) refinement recommendations, and (5) an SVG chart mapping predictions vs. actuals.

## Current Status (Goal ✅ Achieved)
- **Functional pipeline:** `python -m finsight.pipeline` runs end-to-end today, even on locked-down laptops, because it dynamically swaps between a freshly scraped Yahoo Finance dataset (`data/dataset.json`) and the legacy `yfinance`/`pandas` path.
- **LangChain orchestration:** The agents are wired together through `build_langchain_chain()` inside `finsight.pipeline`, so you can reuse the Runnable graph elsewhere or observe it via `scripts/run_langchain_workflow.py`.
- **Chart + explanation:** Every successful run emits `artifacts/predictions_vs_actual.svg` **and** a nicely formatted `report.txt` that mirrors the `AgentOutput` structure (Finance Bro summary, Test Titan verdict, Mr White tutorial, vector-store status, etc.). The SVG now presents a clean two-line comparison (Actual vs. Predicted) for 2020–2022 with proper spacing, axes, annotations, and the MAE/toughest-year callout block beneath the chart.
- **Testing baked in:** Test Titan’s checks are part of the chain; the run only reports success after verifying data coverage, numeric metrics, and chart creation. When dependencies are installed, `pip install -r requirements.txt && python -m finsight.pipeline` demonstrates the full live-data path.

## Agent Overview
| Agent | Responsibility |
| --- | --- |
| **DataCollector** | Pulls (or falls back to cached) S&P 500 data for 2015–2022 and converts it into annual return/volatility metrics. |
| **Research Agent** | Adds macro context for 2015–2019 so later agents understand the regime (e.g., trade wars, Fed shifts). |
| **Prediction Agent** | Learns from 2015–2019 and generates forecasts for 2020–2022 with narrative rationales. |
| **Evaluator** | Compares predictions with actual 2020–2022 results, computes MAE, and flags the hardest year. |
| **Test Titan** | Runs sanity checks on every payload (data coverage, numeric MAE, etc.) and prints `All tests passed ✅` when everything looks good. |
| **Finance Bro** | Explains the evaluator’s findings in plain English and links to the SVG chart. |
| **Refinement Agent** | Suggests concrete improvements (new indicators, crisis detectors, sentiment inputs, …). |
| **Mr White** | Acts as the mentor who walks a passionate student through the architecture and explains how every agent hands off to the next, quoting the live `AgentOutput` state so the story matches the code. |
| **Lord of the Mysteries** | The orchestration layer that keeps the agents in sync and synthesizes the final report. |

## Workflow Summary
1. **Data ingestion:** `collect_sp500_data()` scrapes the Yahoo Finance ^GSPC history page (the same URL you provided) into `data/dataset.json`, then derives annual return/volatility metrics. When scraping is blocked, it falls back to the prior `yfinance`/`pandas` routine and, if necessary, the legacy annual snapshot (`data/sp500_annual_metrics_2015_2022.json`).
2. **Context enrichment:** `research_macro_events()` writes the Research Agent’s 2015–2019 story beats into a Pinecone `finsight-market-events` vector index (or an offline in-memory replica) and hands the retrieved context to `build_prediction_payload()`.
3. **Forecasting:** `prediction_agent()` applies a heuristic baseline plus contextual adjustments to produce 2020–2022 returns and rationales.
4. **Evaluation + charting:** `evaluator_agent()` calculates MAE/error details and `generate_prediction_chart()` saves `artifacts/predictions_vs_actual.svg`, drawing just the 2020–2022 Actual vs. Predicted lines (the true forecast window) with labeled axes, per-point annotations, and a centered legend so the visual matches the written explanation.
5. **Narration + refinement:** `finance_bro_agent()` turns the stats into a conversational summary while `refinement_agent()` proposes upgrades.
6. **Quality gate:** `test_titan_agent()` confirms the dataset, predictions, and evaluation objects are sane before declaring “All tests passed ✅”.
7. **Report packaging:** `run_pipeline()` executes a LangChain `Runnable` sequence that stitches every agent together and saves the consolidated `AgentOutput` as `report.txt`, complete with Mr White’s explainer.

## Dataset Source & Quality
- **Primary feed:** Daily ^GSPC candles scraped directly from the Yahoo Finance history page you linked (period1=1420070400, period2=1668643200). The scraper stores every row as-is inside `data/dataset.json`, so you can open that file and inspect all entries (Date, Open, High, Low, Close, Adj Close, Volume) without running any code.
- **Automated refresh:** `scripts/refresh_dataset.py` re-scrapes the same URL via the pipeline’s helper (`load_history_dataset`) and rewrites both `data/dataset.json` and the derived `data/sp500_annual_metrics_2015_2022.json`, keeping the repository in sync with Yahoo’s feed.
- **Data quality assessment:** The DataCollector converts the scraped daily candles into annual compounded returns plus volatility, then Test Titan checks for missing years or malformed numbers before the rest of the workflow runs. Because the raw dataset is straight from Yahoo Finance—and cached verbatim for reproducibility—we get both fidelity and determinism (with the usual caveat that Yahoo may lag intraday corporate-action adjustments by a day).

### “Proxy blocked download” — what it means
- When you see that warning in the console, it simply indicates that the current machine cannot reach PyPI or Yahoo Finance through its network proxy. FinSight catches that situation, logs it in Mr White’s briefing, and automatically falls back to the cached JSON snapshot so the workflow keeps running.
- The cached file is not “made up”: it was generated from the same Yahoo Finance feed and ships with the repo to keep demos reproducible. You can regenerate it yourself at any time once you regain internet access (see below), which is the best proof of authenticity.
- If you want the live path, install the dependencies after configuring your proxy, e.g. `pip install --proxy http://corp-proxy:8080 -r requirements.txt`, or download the required wheels (`pandas`, `yfinance`, `langchain`, `pinecone`) on an online machine and `pip install /path/to/*.whl` locally.

### Refreshing or validating the dataset yourself
1. Ensure `requests` is installed (already covered by `pip install -r requirements.txt`).
2. Run the helper script to re-scrape Yahoo Finance and overwrite the cached files:
   ```bash
   python scripts/refresh_dataset.py
   ```
3. Inspect `data/dataset.json` (every historical row) or `data/sp500_annual_metrics_2015_2022.json` (the derived annual metrics) to confirm the refresh succeeded.
4. Re-run `python -m finsight.pipeline` and Mr White will cite `data_source: yahoo_history_html` if the scraper succeeded, or fall back to the legacy path if the proxy blocked it.

## LangChain & Pinecone (Mr White’s Lesson)
> “A LangChain Runnable sequence kicks off the show so every agent hands a structured state to the next. The Research Agent ships its macro bullets into Pinecone (or the offline mirror) so later agents can tap a real RAG memory.” – Mr White

- **LangChain orchestration:** `run_pipeline()` is defined as a chain of `RunnableLambda` steps (DataCollector → Research → Prediction → Evaluation → Finance → Refinement → Testing → Mentorship). LangChain keeps the state dictionary flowing so each agent only focuses on its own contract.
- **Runnable source of truth:** The same graph is exposed through `build_langchain_chain()` inside `finsight/pipeline.py`, which means other scripts—or your own experiments—can call the Runnable directly without reimplementing the glue.
- **Offline fallback:** When the real `langchain` package is unavailable (e.g., on a locked-down interview laptop), FinSight automatically swaps in a tiny compatible shim so the same Runnable chain executes without breaking the narrative.
- **Pinecone usage:** `ContextVectorStore` wraps the Pinecone serverless client. When `PINECONE_API_KEY` (and optionally `PINECONE_REGION`, default `us-east-1`) is present, the Research Agent upserts yearly macro summaries into the `finsight-market-events` index and immediately fetches them for downstream agents. Without credentials, the same interface falls back to a deterministic in-memory vector store, but the Mentorship log still reports the Pinecone status so you know which path ran.
- **Status tracking:** Every pipeline run records `vector_store_status` inside `AgentOutput`, and Mr White echoes it in his explanation so interviewers can see whether the live Pinecone integration or the local mock handled the context.
- **CLI overrides:** Don’t want to edit env vars? Pass `--pinecone-api-key` and `--pinecone-region` directly to `python -m finsight.pipeline` (or any helper script) and the run will temporarily use those credentials without touching your shell profile.

To enable the hosted Pinecone path locally, export your keys before running the pipeline:

```bash
export PINECONE_API_KEY="your-key"
# optional if you use a region other than the default us-east-1
export PINECONE_REGION="your-region"
python -m finsight.pipeline
```

## Running FinSight Locally
1. *(Optional but recommended)* Install the lightweight dependencies if you want live data (use `python3 -m pip ...` if your macOS shell maps `python` to Python 2):
   ```bash
   pip install -r requirements.txt
   ```
2. *(Optional for Pinecone)* If you have Pinecone credentials, either set `PINECONE_API_KEY`/`PINECONE_REGION` or pass them inline via `--pinecone-api-key` / `--pinecone-region` so the Research Agent writes to the remote vector index.
3. Execute the pipeline (works even without step 1 thanks to the cached metrics and the in-memory Pinecone fallback):
   ```bash
   python -m finsight.pipeline --pinecone-api-key "$PINECONE_API_KEY"
   ```
4. Open the freshly generated `report.txt` (in the project root) to read the formatted summary, which includes the evaluation table, Finance Bro’s notes, and Mr White’s LangChain + Pinecone explanation. The terminal now simply tells you where the file lives along with Test Titan’s `All tests passed ✅` line.

Need to force a refresh or point at a different Yahoo query? Add `--dataset-url <your-url>`, `--dataset-path /tmp/your_copy.json`, or `--refresh-dataset` to the command above and the scraper will obey those overrides before the agents run.

### Using the Pinecone key you supplied
You shared a live Pinecone Serverless key (`pcsk_4qeazd_6nYxuL7ACTaduh585oVr7mmiFmpSNtdVJYSjAWZbPt44TVUaJVPEJd7LBrdjMFY`). Run the exact command below on macOS to hit the hosted index without touching your shell profile:

```bash
python3 -m finsight.pipeline \
  --pinecone-api-key "pcsk_4qeazd_6nYxuL7ACTaduh585oVr7mmiFmpSNtdVJYSjAWZbPt44TVUaJVPEJd7LBrdjMFY" \
  --pinecone-region us-east-1
```

Prefer environment variables instead? Export `PINECONE_API_KEY` with that value once per terminal session and omit the CLI flags.

### Want to *see* the LangChain orchestration?
- Run the explicit walkthrough script to watch each Runnable stage pass the baton:
 ```bash
  python scripts/run_langchain_workflow.py --pinecone-api-key "$PINECONE_API_KEY"
  ```
- The script uses the same `RunnableLambda` chain under the hood but prints the state keys after every step (bootstrap → collect → context → … → mentor) so you can confirm how LangChain orchestrates the agents in order.
- The output ends with Finance Bro’s summary and the exact SVG file path, proving the application truly ran to completion.

### “pinecone-client” import error (Mac fix)
- The Pinecone project renamed its Python package from `pinecone-client` to `pinecone`. If you see the error `Please remove 'pinecone-client' from your project dependencies and add 'pinecone' instead`, remove the old wheel and install the new one:
  ```bash
  pip uninstall -y pinecone-client
  pip install pinecone
  ```
- After that, re-run `python -m finsight.pipeline` (or `python3 -m ...` on macOS). The pipeline will either connect to Pinecone Serverless when `PINECONE_API_KEY` is set or report `vector_store_status=pinecone_local_memory:import_error:ModuleNotFoundError` if it must fall back to the deterministic local store.

## Viewing the Chart and Results
- **Chart:** After running the pipeline, open `artifacts/predictions_vs_actual.svg` in any browser to see the predicted vs. actual returns for 2020–2022. The refreshed design focuses on two lines (Actual vs. Predicted), keeps the axis labels directly below/along the axes, and adds annotations, a centered legend, and the MAE/toughest-year summary so it matches the requirements for a focused forecast comparison.
- **Agent report:** Every run writes `report.txt` with the structured `AgentOutput` object. It lists the historical metrics, macro context, predictions with rationales, evaluation stats (including MAE and hardest year), Finance Bro’s explanation, refinement ideas, the absolute path to the chart, **and** the `vector_store_status` so you can cite whether Pinecone or the local replica supplied the memories.
- **Reusable data:** The full scraped history lives at `data/dataset.json` (with all rows from the Yahoo page) and the aggregated annual snapshot remains at `data/sp500_annual_metrics_2015_2022.json`, so you can re-run or audit everything offline anytime.

Bring this README to your interview: it covers the pitch, the architecture, and exactly how to reproduce the results—including where to find the visual evidence.
