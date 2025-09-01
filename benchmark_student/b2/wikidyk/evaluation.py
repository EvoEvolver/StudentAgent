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
from narwhals import col
import numpy as np
import pandas as pd
from metrics import metrics
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

XLABEL_MAP = {
    "student": "Student",
    "baseline_agentic" : "AgenticRAG", 
    "baseline_naive": "NaiveRAG", 
    "baseline_pretraining": "NaiveLLM",
    "baseline_answerable": "OracleLLM",
    "Flan-T5-220M": "Flan-T5-220M",
    "Flan-T5-770M": "Flan-T5-770M"
}
COLOR_MAP  = {
    "reliability": "#f5d7b0",
    "generality": "#d15b56",
    "paraphrase": "#c43138",
    "portability": "#7ba8a3",
    "locality": "#3e909d",
    "factual": "#007896", 
    "counterfactual" : "#004e61", 
}

#f5d7b0,  #d15b56,  #c43138,  #7ba8a3,  #3e909d,  #007896 and  #004e61.


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
        if type(correct) == str:
            correct = [correct]
        result = r.get("result")
        output = _safe_extract_output(result)

        computed = {"match": None, "f1": None}
        if output is not None and correct is not None:
            try:
                m = metrics(correct[0], output)  # user-provided
                computed["match"] = 1 if bool(m.get("match")) else 0
                f1val = m.get("f1")
                computed["f1"] = float(f1val)/100 if f1val is not None else None
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

    # --- Sanity checks (row counts) ---
    logging.info("=== Sanity check: row counts ===")
    if not df.empty:
        per_agent = df["agent_id"].value_counts(dropna=False).sort_index()
        logging.info("Rows per agent_id:\n" + per_agent.to_string())
        per_agent_q = (
            df.groupby(["agent_id", "question_type"]).size().rename("rows").sort_index()
        )
        logging.info("Rows per agent_id per question_type:\n" + per_agent_q.to_string())
    else:
        logging.info("No rows loaded; nothing to count.")

    return df


def aggregate_by_agent_qtype(df: pd.DataFrame) -> pd.DataFrame:
    # Count rows with NaN in match/f1
    nan_summary = (
        df[["agent_id", "question_type", "match", "f1"]]
        .assign(match_nan=df["match"].isna(), f1_nan=df["f1"].isna())
        .groupby(["agent_id", "question_type"])
        .agg(
            total_rows=("match", "size"),
            match_nan=("match_nan", "sum"),
            f1_nan=("f1_nan", "sum"),
        )
    )

    print("=== Sanity check: NaN distribution before aggregation ===")
    print("\n" + nan_summary.to_string())

    grp = (
        df.dropna(subset=["match", "f1"])
          .groupby(["agent_id", "question_type"], as_index=False)
          .agg(match_acc=("match", "mean"), f1=("f1", "mean"))
    )
    return grp

def _ordered_qtypes(qtypes):
    return qtypes

def _grouped_bars_ax(ax, pivot_df: pd.DataFrame, ylabel: str):
    """
    Draw grouped bars on ax.
    - x = agents, ordered by XLABEL_MAP
    - bars = qtypes, ordered by COLOR_MAP.keys()
    """
    # order rows (agents) by XLABEL_MAP
    agent_order = [a for a in XLABEL_MAP.keys() if a in pivot_df.index]
    pivot_df = pivot_df.reindex(index=agent_order)

    # order columns (qtypes) by COLOR_MAP
    q_order = [q for q in COLOR_MAP.keys() if q in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=q_order)

    agents = list(pivot_df.index)
    qtypes = list(pivot_df.columns)

    x = np.arange(len(agents))
    n = max(len(qtypes), 1)
    width = 0.8 / n

    for i, q in enumerate(qtypes):
        vals = pivot_df[q].values
        ax.bar(
            x + i * width - (n - 1) * width / 2.0,
            vals,
            width=width,
            color=COLOR_MAP.get(q, None),
            label=q,                 # legend labels = qtype names
            edgecolor="none",
        )

    # x labels from XLABEL_MAP values, in map order
    ax.set_xticks(x, [XLABEL_MAP.get(a, a) for a in agents], rotation=0) # centered labels
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    return qtypes  # for legend construction

def make_plots(agg: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Build the two pivot tables (same as before)
    pivot_match = (
        agg.assign(match_pct=agg["match_acc"] * 100.0)
           .pivot_table(index="agent_id", columns="question_type", values="match_pct", fill_value=0.0)
           .sort_index()
    )
    pivot_f1 = (
        agg.assign(f1_pct=agg["f1"])
           .pivot_table(index="agent_id", columns="question_type", values="f1_pct", fill_value=0.0)
           .sort_index().drop(columns=["factual", "counterfactual"])
    )

    # ----- Combined figure: two rows, one column -----
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=False)

    q_top = _grouped_bars_ax(ax_top, pivot_match, ylabel=r"Match Accuracy / % ($\uparrow$)")
    q_bot = _grouped_bars_ax(ax_bot,  pivot_f1,    ylabel=r"F1 Score ($\uparrow$)")
    # ax_top.set_xticklabels([])

    # Single legend below, ordered by COLOR_MAP.keys()
    q_union = [q.capitalize() for q in COLOR_MAP.keys() if (q in q_top) or (q in q_bot)]
    handles = [mpatches.Patch(facecolor=COLOR_MAP[q.lower()], label=q) for q in q_union]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 7),
        frameon=True,
        title="",
        bbox_to_anchor=(0.5, 0.02),
    )

    # Make space for bottom legend
    plt.subplots_adjust(bottom=0.18, hspace=0.35)

    out_path = os.path.join(outdir, "combined_match_f1_by_agent_vertical.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # Also save CSVs
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
    formatted = wide.map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

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

    RESULTS_PATH = "results/results__benchmark__run__100.jsonl"
    OUTDIR = "output/figures/test__104"
    LATEX_TABLE_PATH = "output/tables/"

    configure_logging()

    rows = load_jsonl(RESULTS_PATH)
    
    # remove rows that have "result": {"error" : {litellm.litellm.InternalServerError*}...}
    rows_removed = []
    for r in rows:
        res = r.get("result", {})
        if type(res) is dict:    
            error = res.get("error", {})
            if error.startswith("litellm.InternalServerError"):
                continue
        rows_removed.append(r)

    rows = rows_removed

    # update overloaded rerun results
    rows_overloaded = load_jsonl("results/results__benchmark__run__100__retry_overload__1756636934.jsonl")
    rows.extend(rows_overloaded)

    
    rows_corrected = load_jsonl("results/results__benchmark__run__101.jsonl")
    rows_rag = load_jsonl("results/results__benchmark__run__104.jsonl")

    # drop all rows with agent_id == "baseline_answerable" and combine rows_corrected into rows
    rows = [r for r in rows if r.get("agent_id") not in ["baseline_answerable","baseline_naive", "baseline_agentic"]]
    rows.extend(rows_corrected)
    rows.extend(rows_rag)

    

    df = compute_metrics_df(rows)
    agg = aggregate_by_agent_qtype(df)

    # After `agg = aggregate_by_agent_qtype(df)`
    check = (
        df.groupby(["agent_id", "question_type"])
        .agg(n=("match", "count"), correct=("match", "sum"))
        .reset_index()
    )
    merged = agg.merge(check, on=["agent_id", "question_type"], how="left")
    merged["match_pct_from_counts"] = merged["correct"] / merged["n"] * 100

    # Log a few rows where the plotted % and recomputed % differ by > 0.01
    delta = (merged["match_acc"] * 100 - merged["match_pct_from_counts"]).abs()
    suspicious = merged[delta > 0.01]
    if not suspicious.empty:
        print("Rows where match % != correct/n by > 0.01:\n%s", suspicious.to_string(index=False))
    else:
        print("All match percentages are consistent with counts.")

    # zero-out F1 for factual/counterfactual after aggregation if desired
    mask_fc = agg["question_type"].isin(["factual", "counterfactual"])
    agg.loc[mask_fc & (agg["f1"].notna()), "f1"] = 0.0

    # add data from WikiDYK paper
    wikidyk_results = [{
        "agent_id": "Flan-T5-220M",
        "reliability": {"match_acc": 56.00, "f1": 58.69},
        "generality": {"match_acc": 34.00, "f1": 33.73},
        "paraphrase": {"match_acc": 47.00, "f1": 49.83},
        "portability": {"match_acc": 21.74, "f1": 28.49},
        "locality": {"match_acc": 20.29, "f1": 20.87},
        "factual": {"match_acc" : 0, "f1": 0},
        "counterfactual": {"match_acc" : 0, "f1": 0},
    },
    {
        "agent_id": "Flan-T5-770M",
        "reliability": {"match_acc": 78.00, "f1": 79.27},
        "generality": {"match_acc": 51.00, "f1": 49.67},
        "paraphrase": {"match_acc": 67.00, "f1": 66.87},
        "portability": {"match_acc": 44.55, "f1": 45.80},
        "locality": {"match_acc": 56.52, "f1": 73.72},
        "factual": {"match_acc" : 0, "f1": 0},
        "counterfactual": {"match_acc" : 0, "f1": 0},
    }]
    rows = []
    for res in wikidyk_results[1:]:
        agent_id = res["agent_id"]
        for qtype, metrics in res.items():
            if qtype == "agent_id":
                continue
            rows.append({
                "agent_id": agent_id,
                "question_type": qtype,
                "match_acc": metrics["match_acc"]/100,
                "f1": metrics["f1"]/100
            })

    agg = pd.concat([agg, pd.DataFrame(rows)])

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
