'''
WikiDYK (n = 50 facts)
- Learning:  Multiple Facts (+ Single Fact)
- Evaluation:
    - Baselines:
        - answerable ~ LLM + fact + question + answer
        - pretraining ~ LLM + question
        - LLM + fact + question
        - LLM + RAG@learning + question
        - LLM + RAG + question
        
    - Metrics:
        - Accuracy overall
        - Accuracy, if answerable
        - Per question style: boxplot (mean, std, median)
        - Per fact: histogram (mean)
'''

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from student.agent.agent_student import StudentAgent
from student.agent.rag_agent import RAGAgent
from mllm import Chat
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results/results_wikidyk_all"
os.makedirs(RESULTS_DIR, exist_ok=True)
MAX_WORKERS = 4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename='logs/wikidyk_all_eval.log',
    filemode='a'
)

BASELINES = [
    "answerable",      # LLM + fact + question + answer
    "pretraining",     # LLM + question
    "fact_question",   # LLM + fact + question
    "rag_learning",    # LLM + RAG@learning + question
    "rag"              # LLM + RAG + question
]


def eval_answer(question, answer, correct_answer) -> Tuple[str, str]:
    prompt = f"""
    You are an evaluator for a knowledge test. 
    Your task is to decide if the <test_answer> is fully correct, given the <question> and the <correct_answer>. 
    Use only the information provided. Do not make assumptions beyond the given answers.

    Inputs:
    <question>
    {question}
    </question>

    <test_answer>
    {answer}
    </test_answer>

    <correct_answer>
    {correct_answer}
    </correct_answer>

    Instructions:
    - Output CORRECT if <test_answer> answers the question as good as <correct_answer>.
    - Output INCORRECT if <test_answer> is incorrect or does not sufficiently match <correct_answer>.

    Example outputs (IMPORTANT: never start anyway different):
    CORRECT (+ explaination)
    INCORRECT (+ explaination)
    
    """
    chat = Chat(dedent=True)
    chat += prompt
    res = chat.complete(cache=False, expensive=True).strip()
    if res.startswith("CORRECT"):
        return "CORRECT", ""
    elif res.startswith("INCORRECT"):
        return "INCORRECT", res
    else:
        return "ERROR", res


def setup_agent(baseline_type, fact=None):
    """
    Returns the correct agent instance for the baseline.
    For RAG agents, initializes with different doc.txt paths.
    """
    if baseline_type in ["answerable", "pretraining", "fact_question"]:
        agent = StudentAgent(provider="anthropic")
        if fact:
            agent.run(fact)
        return agent
    elif baseline_type == "rag_learning":
        # RAGAgent with learning docs
        return RAGAgent(doc_path="docs/learning_doc.txt")
    elif baseline_type == "rag":
        # RAGAgent with general docs
        return RAGAgent(doc_path="docs/general_doc.txt")
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def run_baseline_experiment(baseline_type, fact, qa, run_id, fact_id, eval_type):
    agent = setup_agent(baseline_type, fact)
    identifier = f"{run_id}__{fact_id}__{eval_type}__{baseline_type}"
    out_file = os.path.join(RESULTS_DIR, f"{identifier}.jsonl")
    question = qa["prompt"]
    correct_answer = qa["answer"][0] if isinstance(qa["answer"], list) else qa["answer"]
    prediction = ""
    try:
        prediction = agent.run(question)
    except Exception as e:
        logging.error(f"{identifier} failed on question: {question} | {e}")
        prediction = ""
    try:
        verdict, explanation = eval_answer(question, prediction, correct_answer)
    except Exception as e:
        verdict, explanation = "ERROR", str(e)
    result = {
        "identifier": identifier,
        "fact_id": fact_id,
        "eval_type": eval_type,
        "run_id": run_id,
        "baseline": baseline_type,
        "question": question,
        "correct_answer": correct_answer,
        "fact": fact,
        "prediction": prediction,
        "verdict": verdict,
        "explanation": explanation,
    }
    with open(out_file, "w") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def collect_metrics(results: List[Dict]) -> Dict:
    metrics = {}
    verdicts = [r["verdict"] for r in results]
    correct = [v == "CORRECT" for v in verdicts]
    metrics["accuracy_overall"] = np.mean(correct)
    answerable = [r for r in results if r["baseline"] == "answerable"]
    metrics["accuracy_answerable"] = np.mean([r["verdict"] == "CORRECT" for r in answerable]) if answerable else None
    # Per question style
    per_question = {}
    for r in results:
        q = r["eval_type"]
        per_question.setdefault(q, []).append(r["verdict"] == "CORRECT")
    metrics["per_question_style"] = {k: {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v))
    } for k, v in per_question.items()}
    # Per fact
    per_fact = {}
    for r in results:
        f = r["fact_id"]
        per_fact.setdefault(f, []).append(r["verdict"] == "CORRECT")
    metrics["per_fact_histogram"] = {k: float(np.mean(v)) for k, v in per_fact.items()}
    return metrics


def load_all_metrics(results_dir=RESULTS_DIR):
    """Loads all result jsonl files and returns a list of dicts."""
    results = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(results_dir, fname), "r") as f:
                for line in f:
                    results.append(json.loads(line))
    return results


def metrics_table(metrics):
    """Prints a summary table of metrics."""
    print("\nMETRICS SUMMARY TABLE:")
    print(f"Accuracy overall: {metrics['accuracy_overall']:.3f}")
    print(f"Accuracy (answerable): {metrics['accuracy_answerable']:.3f}")
    print("Per question style:")
    for k, v in metrics["per_question_style"].items():
        print(f"  {k}: mean={v['mean']:.3f}, std={v['std']:.3f}, median={v['median']:.3f}")
    print("Per fact histogram:")
    for k, v in metrics["per_fact_histogram"].items():
        print(f"  fact_id {k}: mean={v:.3f}")


def plot_metrics(metrics):
    """Plots boxplot for per question style and histogram for per fact."""
    # Boxplot per question style
    question_data = [(k, v['mean']) for k, v in metrics['per_question_style'].items()]
    df_q = pd.DataFrame(question_data, columns=["question_style", "accuracy"])
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="question_style", y="accuracy", data=df_q)
    plt.title("Accuracy per Question Style")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # Histogram per fact
    fact_data = [(k, v) for k, v in metrics['per_fact_histogram'].items()]
    df_f = pd.DataFrame(fact_data, columns=["fact_id", "accuracy"])
    plt.figure(figsize=(8, 4))
    sns.histplot(df_f["accuracy"], bins=20)
    plt.title("Accuracy per Fact (Histogram)")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def main():
    wikidyk = pd.read_parquet("hf://datasets/YWZBrandon/wikidyk/data/test-00000-of-00001.parquet")
    data = wikidyk[["fact", "eval"]].drop_duplicates()
    run_id = "wikidyk_all"
    jobs = []
    for i, row in data[:50].iterrows():
        fact_id = str(i)
        fact = row["fact"]
        evals = json.loads(row["eval"]) if isinstance(row["eval"], str) else row["eval"]
        for eval_type, qa in evals.items():
            for baseline in BASELINES:
                jobs.append((baseline, fact, qa, run_id, fact_id, eval_type))
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_baseline_experiment, *args) for args in jobs]
        for f in as_completed(futures):
            try:
                result = f.result()
                results.append(result)
                logging.info(f"{result['identifier']}: complete")
            except Exception as e:
                logging.error(f"Experiment failed: {e}")
    metrics = collect_metrics(results)
    metrics_table(metrics)
    plot_metrics(metrics)

if __name__ == "__main__":
    main()