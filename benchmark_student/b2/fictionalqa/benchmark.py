import pandas as pd
import json
import logging
import time
from datetime import datetime, timezone
from filelock import FileLock
from student.agent import *
from student.agent.agent_baselines import BaselineAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger("benchmark")

training_run_id = "002"
RAG_MEMORY_PATH = f"memory/combined_fqa_rag_{training_run_id}.parquet"
STUDENT_MEMORY_PATH = f"checkpoints/memory_{training_run_id}"


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
SETUP_PATH = f"setup/setups_fqa_{training_run_id}.json"


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
        self.event_id = exp_args["event_id"]
        self.fiction_id = exp_args["fiction_id"]
        self.context = exp_args["context"]
        self.question_id = exp_args["question_id"]
        self.question = exp_args["question"]

        self.target = exp_args["target"]
        self.topk_choices = exp_args["topk_choices"]
        self.result = None

    @classmethod
    def from_dict(cls, d):
        required = [
            "agent_id", "run_id", "event_id", "question_id",
            "question", "target", "topk_choices", "context"
        ]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        return cls(d)

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "fiction_id": self.fiction_id,
            
            "context": self.context,

            "question_id": self.question_id,
            "question": self.question,
            
            "target": self.target,
            "topk_choices" : self.topk_choices,
            "result": self.result,
        }

    # ---- Internal helpers ----
    def identifier(self):
        return f"{self.run_id}__{self.question_id}__{self.agent_id}"

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

        prompt = f"Question: {self.question}\nPossible choices: {self.topk_choices}\nAnswer (MUST BE ONE OF THE CHOICES!): "

        try:
            if self.agent_id in ["baseline_naive", "student", "baseline_agentic"]:
                return agent.run(prompt)

            elif self.agent_id == "baseline_answerable":
                return agent.run_answerable(context=self.context, question=self.question)

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
        token_count = agent.reset_token_count()

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
            "token_count": token_count
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
        exp_id = f"{exp_cfg.get('run_id')}__{exp_cfg.get('question_id')}__{exp_cfg.get('agent_id')}"
        logger.exception(f"[{exp_id}] Unhandled failure in parallel worker: {e}")
        return {
            "agent_id": exp_cfg.get("agent_id"),
            "run_id": exp_cfg.get("run_id"),
            "event_id": exp_cfg.get("event_id"),
            "context": exp_cfg.get("context"),
            "question_id": exp_cfg.get("question_id"),
            "question": exp_cfg.get("question"),
            "target": exp_cfg.get("target"),
            "topk_choices": exp_cfg.get("topk_choices"),
            "result": {"error": f"Unhandled: {str(e)}"},
            "_meta": {"experiment_id": exp_id, "parallel_error": True},
            "token_count" : {"input_tokens": 0, "output_tokens": 0}
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

def load_setups():
    with open(SETUP_PATH, "r") as f:
        setups = json.load(f)
    return setups

def setup_experiment():
    experiments = []
    
    setups = load_setups()

    for event_id in setups.keys():
        for setup in setups[event_id]:
            for agent_id in AGENT_IDS:
                
                exp = {
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "event_id": setup["event_id"],
                    "context": setup["context"],
                    "fiction_id": setup["fiction_id"],
                    "question_id": setup["question_id"],
                    "question": setup["question"],
                    "target": setup["target"],
                    "topk_choices": setup["topk_choices"],
                }
                experiments.append(exp)
    
    return experiments

if __name__ == "__main__":
    
    n_fqa = 1
    run_id = "test__001"
    configure_logging(level=logging.INFO, log_file=f"logs/benchmark__{run_id}.log")


    experiments = setup_experiment()

    summary = run_experiments_parallel(
        experiments,
        outfile=f"results/results__{run_id}.jsonl",
        max_workers=8,           
        per_task_timeout=None,  
    )

    print("Summary:", {k: v for k, v in summary.items() if k != "results_sample" and k != "failures"})