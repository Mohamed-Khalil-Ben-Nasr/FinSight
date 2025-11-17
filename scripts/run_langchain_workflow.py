"""Run the FinSight pipeline via an explicit LangChain Runnable chain."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finsight.pipeline import AgentOutput, build_langchain_chain, state_to_agent_output


def main() -> None:
    stage_log = []

    def log_stage(name: str, state: Dict[str, object]) -> None:
        keys = ", ".join(sorted(state.keys()))
        stage_log.append(name)
        print(f"[{name}] state keys -> {keys}")

    chain = build_langchain_chain(stage_logger=log_stage)
    final_state = chain.invoke({})
    output: AgentOutput = state_to_agent_output(final_state)

    print("\nLangChain stages executed (in order):", " -> ".join(stage_log))
    print("Finance Bro summary:\n", output.finance_bro_summary)
    print("Chart saved at:", output.chart_path)

    if output.chart_path:
        path = Path(output.chart_path)
        if path.exists():
            print(f"Chart file size: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
