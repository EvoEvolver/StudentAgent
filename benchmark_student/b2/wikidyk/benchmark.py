import pandas as pd
import json
import logging
import time
from datetime import datetime, timezone
from filelock import FileLock
from student.agent import *
from baselines import BaselineAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger("benchmark")

# ---- Constants ----
RAG_MEMORY_PATH = "memory/wikidyk_rag.parquet"
STUDENT_MEMORY_PATH = "checkpoints/training_003"
AGENT_CONFIG = {
    "expensive": False,
    "provider": "anthropic",
    "cache": False,
}
AGENT_IDS = [
    "baseline_naive",
    "baseline_agentic",
    "student",
    "baseline_pretraining",
    "baseline_answerable",
]

# ---- Logging setup (call configure_logging() once in your main) ----
def configure_logging(level=logging.INFO, log_file=None):
    """
    Configure root logger for console + optional file.
    Uses a simple, readable format with timestamps.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet noisy deps if any


class Experiment:
    def __init__(self, exp_args):
        # Required fields
        self.agent_id = exp_args["agent_id"]
        self.run_id = exp_args["run_id"]
        self.fact_id = exp_args["fact_id"]
        self.fact = exp_args["fact"]
        self.question_type = exp_args["question_type"]
        self.question = exp_args["question"]
        self.correct_answer = exp_args["correct_answer"]

        # Will be filled after running
        self.result = None

    @classmethod
    def from_dict(cls, d):
        required = [
            "agent_id", "run_id", "fact_id", "fact",
            "question_type", "question", "correct_answer"
        ]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        return cls(d)

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "fact_id": self.fact_id,
            "fact": self.fact,
            "question_type": self.question_type,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "result": self.result,
        }

    # ---- Internal helpers ----
    def identifier(self):
        return f"{self.run_id}__{self.fact_id}__{self.question_type}__{self.agent_id}"

    def _utc_now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _safe_append_jsonl(self, file_name, data):
        lock = FileLock(file_name + ".lock")
        with lock:
            with open(file_name, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ---- Agent plumbing ----
    def get_agent(self):
        try:
            if self.agent_id == "baseline_naive":
                return NaiveRAGAgent(memory_path=RAG_MEMORY_PATH, **AGENT_CONFIG)

            elif self.agent_id == "baseline_agentic":
                return AgenticRAGAgent(memory_path=RAG_MEMORY_PATH, **AGENT_CONFIG)

            elif self.agent_id == "student":
                student = StudentAgent(**AGENT_CONFIG)
                student.load(STUDENT_MEMORY_PATH)
                student.setup_quiz()
                return student

            elif self.agent_id in ["baseline_pretraining", "baseline_answerable"]:
                return BaselineAgent(**AGENT_CONFIG)

            else:
                raise ValueError(f"Invalid agent id: {self.agent_id}")

        except Exception as e:
            logger.exception(f"[{self.identifier()}] Failed to init agent: {e}")
            return None

    def run_agent(self, agent):
        if agent is None:
            return {"error": "Agent init failed"}

        prompt = f"Question: {self.question}\nAnswer: "

        try:
            if self.agent_id in ["baseline_naive", "student", "baseline_agentic"]:
                return agent.run(prompt)

            elif self.agent_id == "baseline_answerable":
                return agent.run_answerable(context=self.fact, question=self.question)

            elif self.agent_id == "baseline_pretraining":
                return agent.run_pretraining(question=self.question)

            else:
                return {"error": f"Invalid agent id at runtime: {self.agent_id}"}

        except Exception as e:
            logger.exception(f"[{self.identifier()}] Agent run failed: {e}")
            return {"error": str(e)}

    def run_experiment(self, file_name="experiments.jsonl"):
        """
        Runs the agent and appends a single JSON record to file_name.
        Safe for concurrent use (file lock).
        """
        exp_id = self.identifier()
        start_time = self._utc_now_iso()
        t0 = time.perf_counter()

        logger.info(f"Starting experiment: {exp_id}")

        agent = self.get_agent()
        response = self.run_agent(agent)
        self.result = response

        duration_sec = round(time.perf_counter() - t0, 4)
        end_time = self._utc_now_iso()

        # Enrich record with metadata for benchmarking
        record = self.to_dict()
        record["_meta"] = {
            "experiment_id": exp_id,
            "started_at": start_time,
            "finished_at": end_time,
            "duration_sec": duration_sec,
            "agent_config": AGENT_CONFIG,  # snapshot for traceability
            "write_file": file_name,
        }

        # Write (append) safely
        try:
            self._safe_append_jsonl(file_name, record)
            logger.info(f"Finished experiment: {exp_id} in {duration_sec}s -> written to {file_name}")
        except Exception as e:
            logger.exception(f"[{exp_id}] Failed to write results: {e}")
            # still keep it visible to caller
            return record

        return record

    @classmethod
    def from_json(cls, file_name: str):
        items = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(cls.from_dict(json.loads(line)))
        return items


def _run_single_experiment(exp_cfg: Dict[str, Any], outfile: str) -> Dict[str, Any]:
    try:
        exp = Experiment(exp_cfg)
        return exp.run_experiment(file_name=outfile)
    except Exception as e:
        # Hard failure constructing/running experiment
        # (run_experiment already logs; this captures construction-time issues)
        exp_id = f"{exp_cfg.get('run_id')}__{exp_cfg.get('fact_id')}__{exp_cfg.get('question_type')}__{exp_cfg.get('agent_id')}"
        logger.exception(f"[{exp_id}] Unhandled failure in parallel worker: {e}")
        return {
            "agent_id": exp_cfg.get("agent_id"),
            "run_id": exp_cfg.get("run_id"),
            "fact_id": exp_cfg.get("fact_id"),
            "fact": exp_cfg.get("fact"),
            "question_type": exp_cfg.get("question_type"),
            "question": exp_cfg.get("question"),
            "correct_answer": exp_cfg.get("correct_answer"),
            "result": {"error": f"Unhandled: {str(e)}"},
            "_meta": {"experiment_id": exp_id, "parallel_error": True},
        }

def run_experiments_parallel(
    experiments: List[Dict[str, Any]],
    outfile: str = "experiments.jsonl",
    max_workers: Optional[int] = None,
    per_task_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Runs many experiments in parallel, appending each result to `outfile`.
    Returns a summary dict (counts + failures captured).
    - max_workers: defaults to sensible CPU*5 for IO-bound if None.
    - per_task_timeout: optional timeout (seconds) per experiment.
    """
    if max_workers is None:
        # For IO-bound workloads (API calls), use a wider pool.
        import os
        cpu = os.cpu_count() or 4
        max_workers = min(64, cpu * 5)

    logger.info(f"Launching {len(experiments)} experiments with max_workers={max_workers}; output -> {outfile}")

    completed = 0
    failed = 0
    results_sample = []  # keep a small sample for quick inspection
    failures: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_experiment, cfg, outfile): cfg for cfg in experiments
        }
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                record = fut.result(timeout=per_task_timeout)
                completed += 1
                # quick sniff test for errors
                if isinstance(record, dict) and isinstance(record.get("result"), dict) and record["result"].get("error"):
                    failed += 1
                    failures.append({"cfg": cfg, "error": record["result"]["error"]})
                # keep at most 10 for quick preview
                if len(results_sample) < 10:
                    results_sample.append(record)
            except Exception as e:
                failed += 1
                logger.exception(f"Task crashed: {e}")
                failures.append({"cfg": cfg, "error": str(e)})

    summary = {
        "total": len(experiments),
        "completed": completed,
        "failed": failed,
        "outfile": outfile,
        "results_sample": results_sample,  # first few records
        "failures": failures[:10],         # first few failures
    }
    logger.info(
        f"Parallel run summary: total={summary['total']} completed={summary['completed']} failed={summary['failed']} -> {outfile}"
    )
    return summary


if __name__ == "__main__":
    n_wikidyk = 5

    wikidyk = pd.read_parquet("hf://datasets/YWZBrandon/wikidyk/data/test-00000-of-00001.parquet")
    wikidyk_data = wikidyk[["fact", "eval"]].drop_duplicates()
    wikidyk_data_sampled =wikidyk_data.sample(n=n_wikidyk, random_state=10)
    
    run_id = "test_run_001"
    
    configure_logging(level=logging.INFO, log_file=f"logs/benchmark__{run_id}.log")

    experiments = []
    for i, row in wikidyk_data_sampled[:2].iterrows():
        fact_id = str(i)
        fact = row["fact"]
        evals = json.loads(row["eval"]) if isinstance(row["eval"], str) else row["eval"]
        for question_type, qa in evals.items():
            question = qa["prompt"]
            correct_answer = qa["answer"]
            for agent_id in AGENT_IDS:
                
                exp = {
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "fact_id": str(i),
                    "fact": fact,
                    "question_type": question_type,
                    "question": question,
                    "correct_answer": correct_answer,
                }
                experiments.append(exp)


    summary = run_experiments_parallel(
        experiments,
        outfile=f"results/results__{run_id}.jsonl",
        max_workers=8,           
        per_task_timeout=None,   # or e.g. 120 for 2 min per task
    )

    print("Summary:", {k: v for k, v in summary.items() if k != "results_sample" and k != "failures"})