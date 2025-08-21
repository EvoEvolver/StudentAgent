"""
Analysis script for RAG benchmarking results stored as JSON Lines (JSONL).
Inputs
------
- One JSONL file (default: experiments.jsonl) with one record per experiment.
  Each record should contain at least:
    agent_id, run_id, fact_id, fact, question_type, question, correct_answer, result
  Optionally a metadata key `_meta` with timing fields.

Metrics
-------
Expects a function `metrics(correct_answer, model_output)` to be importable
(from a local `metrics.py`). It must return a dict with:
    {"match": bool, "f1": float}
where:
    - "match": True/False (will be converted to 1/0)
    - "f1": a float in [0, 1]

What this script does
---------------------
1) Loads the JSONL and computes metrics for each row (robust to various result shapes).
2) Aggregates by [agent_id, question_type] computing:
    - match accuracy = mean(match)
    - f1 average = mean(f1)
3) Saves two grouped-bar plots (per agent_id, bars = question types):
    - match accuracy (%)
    - f1 score (%)
4) Creates a LaTeX-ready table DataFrame with a column layout similar to the example:
       Model | Obj. | <QType1: Match | F1> | <QType2: Match | F1> | ...
   and optionally writes it to a .tex file.

Usage
-----
python analysis.py \
  --input experiments.jsonl \
  --outdir ./analysis_out \
  --latex-table latex_table.tex \
  --agent-map agent_map.json

Notes
-----
- The LaTeX table tries to parse (Model, Obj.) from `agent_id` via a mapping file.
  Provide a JSON file like:
      {
        "baseline_naive": {"Model": "Llama-3-2-8B", "Obj": "QA"},
        "baseline_agentic": {"Model": "Llama-3-2-8B", "Obj": "SP"},
        "student": {"Model": "Flan-T5", "Obj": "SP"}
      }
  If no mapping is provided, we set Model = agent_id and Obj = "".
- Plots use matplotlib only (no seaborn) and do not set custom colors.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics import metrics

def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _safe_extract_output(result: Any) -> Optional[str]:
    if result is None:
        return None
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        # common patterns
        if "error" in result and result["error"]:
            return None
        for key in ("text", "output", "answer", "response", "content"):
            if key in result and isinstance(result[key], str):
                return result[key].strip()
        # last resort: stringify
        return json.dumps(result)
    # fallback
    return str(result)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            ln = line.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception as e:
                logging.warning(f"Skipping bad JSON line {i}: {e}")
                continue
    return rows


def compute_metrics_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []
    for r in rows:
        agent_id = r.get("agent_id")
        qtype = r.get("question_type")
        correct = r.get("correct_answer")
        result = r.get("result")
        output = _safe_extract_output(result)

        computed = {"match": None, "f1": None}
        if output is not None and correct is not None:
            try:
                m = metrics(correct[0], output)  # user-provided
                computed["match"] = 1 if bool(m.get("match")) else 0
                f1val = m.get("f1")
                computed["f1"] = float(f1val) if f1val is not None else None
            except Exception as e:
                logging.warning(f"metrics() failed for agent_id={agent_id}, qtype={qtype}: {e}")

        base = {
            "agent_id": agent_id,
            "run_id": r.get("run_id"),
            "fact_id": r.get("fact_id"),
            "question_type": qtype,
            "question": r.get("question"),
            "correct_answer": correct,
            "model_output": output,
        }
        base.update(computed)
        records.append(base)

    df = pd.DataFrame.from_records(records)
    return df


def aggregate_by_agent_qtype(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.dropna(subset=["match", "f1"])
          .groupby(["agent_id", "question_type"], as_index=False)
          .agg(match_acc=("match", "mean"), f1=("f1", "mean"))
    )
    return grp


def _ordered_qtypes(unique_qtypes: List[str]) -> List[str]:
    # Preferred order; fall back to whatever exists
    preferred = ["Reliability", "Generality", "Paraphrase", "Portability", "Locality"]
    upper_map = {q.upper(): q for q in unique_qtypes}
    ordered = [upper_map[q.upper()] for q in preferred if q.upper() in upper_map]
    # include any remaining qtypes not in preferred
    for q in unique_qtypes:
        if q not in ordered:
            ordered.append(q)
    return ordered


def _grouped_bar_plot(pivot_df: pd.DataFrame, title: str, ylabel: str, out_path: str):
    # pivot_df: index=agent_id, columns=question_type, values = metric (%)
    agents = list(pivot_df.index)
    qtypes = list(pivot_df.columns)

    x = np.arange(len(agents))
    n = len(qtypes)
    width = 0.8 / max(n, 1)  # keep total width reasonable

    plt.figure(figsize=(10, 4))
    for i, q in enumerate(qtypes):
        vals = pivot_df[q].values
        plt.bar(x + i * width - (n - 1) * width / 2.0, vals, width=width, label=q)

    plt.xticks(x, agents, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Question Type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_plots(agg: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Order columns (qtypes) consistently
    q_order = _ordered_qtypes(sorted(agg["question_type"].dropna().unique().tolist()))

    # Match accuracy (%)
    pivot_match = (
        agg.assign(match_pct=agg["match_acc"] * 100.0)
           .pivot_table(index="agent_id", columns="question_type", values="match_pct", fill_value=0.0)
           .reindex(columns=[q for q in q_order if q in agg["question_type"].unique()])
           .sort_index()
    )
    _grouped_bar_plot(
        pivot_match,
        title="Match Accuracy by Agent (higher is better)",
        ylabel="Match Accuracy (%)",
        out_path=os.path.join(outdir, "match_accuracy_by_agent.png"),
    )

    # F1 (%)
    pivot_f1 = (
        agg.assign(f1_pct=agg["f1"])
           .pivot_table(index="agent_id", columns="question_type", values="f1_pct", fill_value=0.0)
           .reindex(columns=[q for q in q_order if q in agg["question_type"].unique()])
           .sort_index()
    )
    _grouped_bar_plot(
        pivot_f1,
        title="F1 by Agent (higher is better)",
        ylabel="F1 (%)",
        out_path=os.path.join(outdir, "f1_by_agent.png"),
    )

    # also save CSVs for convenience
    pivot_match.to_csv(os.path.join(outdir, "match_accuracy_by_agent.csv"))
    pivot_f1.to_csv(os.path.join(outdir, "f1_by_agent.csv"))


def _parse_agent_to_model_obj(agent_id: str, amap: Optional[Dict[str, Dict[str, str]]] = None):
    if amap and agent_id in amap:
        md = amap[agent_id]
        return md.get("Model", agent_id), md.get("Obj", "")
    # default: leave as-is
    return agent_id, ""


def build_latex_table_df(agg: pd.DataFrame, agent_map: Optional[Dict[str, Dict[str, str]]] = None) -> pd.DataFrame:
    # Prepare percentages
    df = agg.copy()
    df["Match"] = df["match_acc"] * 100.0
    df["F1"] = df["f1"] * 100.0

    # Identify question types in preferred order
    q_order = _ordered_qtypes(sorted(df["question_type"].dropna().unique().tolist()))

    # Build a wide table per agent with two columns (Match, F1) for each qtype
    # First, reshape so we can pivot both metrics
    melted = df.melt(
        id_vars=["agent_id", "question_type"],
        value_vars=["Match", "F1"],
        var_name="metric",
        value_name="value",
    )

    wide = (
        melted.pivot_table(
            index="agent_id",
            columns=["question_type", "metric"],
            values="value",
            aggfunc="mean",
        )
        .reindex(columns=pd.MultiIndex.from_product([q_order, ["Match", "F1"]]))
        .sort_index()
    )

    # Convert to a regular DataFrame with Model/Obj. up front
    model_obj_rows = [ _parse_agent_to_model_obj(aid, agent_map) for aid in wide.index ]
    model_col = [m for (m, _) in model_obj_rows]
    obj_col = [o for (_, o) in model_obj_rows]

    # Format as 2 decimal strings
    formatted = wide.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

    # Build final frame
    formatted.insert(0, ("_", "Obj."), obj_col)     # temporary prefix for stable order
    formatted.insert(0, ("_", "Model"), model_col)
    formatted.columns = pd.MultiIndex.from_tuples(formatted.columns)

    # Sort columns: Model, Obj., then qtypes x (Match,F1)
    cols = [("_", "Model"), ("_", "Obj.")] + [(q, sub) for q in q_order for sub in ("Match", "F1")]
    formatted = formatted.reindex(columns=pd.MultiIndex.from_tuples(cols))

    # Remove the helper top-level "_" for the first two columns by flattening afterwards
    formatted.columns = pd.MultiIndex.from_tuples(
        [("Model", "") if c == ("_", "Model") else
         ("Obj.", "") if c == ("_", "Obj.") else c for c in formatted.columns]
    )

    # Reset index to turn agent_id into rows (will be replaced by Model/Obj.)
    formatted = formatted.reset_index(drop=True)

    return formatted


def save_latex_table(df_latex: pd.DataFrame, path: str):
    # Use to_latex with MultiIndex columns, no escaping (we aren't adding LaTeX chars here)
    with open(path, "w", encoding="utf-8") as f:
        f.write(df_latex.to_latex(index=False, escape=False, multicolumn=True, multicolumn_format='c'))


def main():

    RESULTS_PATH = "results/results__test_run_001.jsonl"
    OUTDIR = "output/figures/test_001"
    LATEX_TABLE_PATH = "output/tables/"

    configure_logging()

    rows = load_jsonl(RESULTS_PATH)

    df = compute_metrics_df(rows)
    agg = aggregate_by_agent_qtype(df)
    

    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(os.path.join(OUTDIR, "per_example_with_metrics.csv"), index=False)
    agg.to_csv(os.path.join(OUTDIR, "agg_by_agent_qtype.csv"), index=False)
    make_plots(agg, OUTDIR)

    # LaTeX export
    agent_map = None

    latex_df = build_latex_table_df(agg, agent_map=agent_map)
    latex_df.to_csv(os.path.join(LATEX_TABLE_PATH, "latex_table_preview.csv"), index=False)

    os.makedirs(os.path.dirname(LATEX_TABLE_PATH), exist_ok=True)
    save_latex_table(latex_df, os.path.join(LATEX_TABLE_PATH, "latex_table.tex"))

    logging.info(f"Analysis complete. Outputs in: {OUTDIR}")
    if LATEX_TABLE_PATH:
        logging.info(f"LaTeX table saved to: {LATEX_TABLE_PATH}")


if __name__ == "__main__":
    main()
