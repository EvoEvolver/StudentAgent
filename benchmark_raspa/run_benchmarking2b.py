import time
from utils import *
import os
import random
import json
from student.agent.agent_raspa import RaspaAgent
random.seed(92)
from student.session_manager import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from logging.handlers import RotatingFileHandler
import threading
import tempfile

import litellm
litellm._turn_on_debug()

import dotenv
dotenv.load_dotenv()


run_id = "b1_matching"
benchmarking_id = "benchmark_2b"
i = 3

ads_dil = "Determine the adsorption enthalpy of {molecule} on {framework} using a simulation at infinite dilution"
ads_1 = "Determine the adsorption enthalpy of {molecule} on {framework} at a pressure of 1e4 and 300 Kelvin"
ads_2 = "Compare the adsorption enthalpies of a 1:1 mixture of {molecule} and {molecule2} on {framework} at a pressure of 1e4 and 300 Kelvin"
h = "Determine the henry coefficient of {molecule} on {framework}"
h_2 = "Determine the henry coefficient of {molecule} and {molecule2} on {framework}"

tasks_multistep = [ads_dil, ads_1, ads_2, h, h_2]


add_hvf = " given the helium void fraction of {hvf}"
add_rb_1 = " and the ideal gas rosenbluth weight of {rosenbluth} for {molecule}"
add_rb_2 = " and the ideal gas rosenbluth weight of {rosenbluth} for {molecule} and {rosenbluth2} for {molecule2}"

hvf = "Calculate the helium void fraction of {framework}"
surface = "Determine the surface area of {framework}"
rosenbluth_1 = "Calculate the ideal Rosenbluth weights for {molecule}"
rosenbluth_2 = "Calculate the ideal Rosenbluth weights for {molecule} and {molecule2}"

tasks_framework = [hvf, surface]                                    # framework
tasks_n1 = [rosenbluth_1]                                           # molecule
tasks_n2 = [rosenbluth_2]                                           # molecule, molecule2

tasks_n1_s = [i + add_hvf for i in [ads_dil, ads_1, h]]             # molecule, framework, hvf
tasks_n1_l = [i + add_hvf + add_rb_1 for i in [ads_dil, ads_1, h]]  # molecule, framework, hvf

tasks_n2_ss = [i + add_hvf for i in [ads_2, h_2]]                   # molecule, molecule2, framework, hvf
tasks_n2_sl = [i + add_hvf + add_rb_1 for i in [ads_2, h_2]]                   # molecule, molecule2, framework, hvf
tasks_n2_ll = [i + add_hvf + add_rb_2 for i in [ads_2, h_2]]                   # molecule, molecule2, framework, hvf

molecules_s = ["CO2", "N2", "methane", "ethane"]
molecules_l = ["n-pentane", "n-hexane", "n-heptane"]
rosenbluth = ["0.0197439", "0.0029442", "0.0004450"] # from Aastha

framework = "IRMOF-13"
f_hvf = 0.877

instructions_multi = ["Explain all the full procedure to solve this task with your tools (WITHOUT DOING IT): ", "Answer this question using simulations (ALWAYS USE 1/10 cycles and up to 10 molecules for speed. IGNORE the low accuracy!): "]
instructions_single = ["Setup the simulation (WITHOUT EXECUTING IT): ", "Answer this question using simulations (ALWAYS USE 1/10 cycles and up to 10 molecules for speed. IGNORE the low accuracy!): "]

def task_prompt(instruction, task, parameters):
    return instruction + task.format(**parameters)

def task_parameters(r, small=True, mixed = True):

    random.seed(r)
    rx = random.sample(range(len(molecules_l)), 2)

    if mixed is True:
        m1 = molecules_l[rx[0]]
        m2 = molecules_s[rx[1]]
        r1 = rosenbluth[rx[0]]

        parameters = {
            "framework" : framework,
            "hvf" : f_hvf,
            "molecule" : m1,
            "molecule2" : m2,
            "rosenbluth" : r1,
        }
    else:
        if small is True:

            m1 = molecules_s[rx[0]]
            m2 = molecules_s[rx[1]]

            parameters = {
                "framework" : framework,
                "hvf" : f_hvf,
                "molecule" : m1,
                "molecule2" : m2,
            }

        else:

            m1 = molecules_l[rx[0]]
            m2 = molecules_l[rx[1]]
            r1 = rosenbluth[rx[0]]
            r2 = rosenbluth[rx[1]]

            parameters = {
                "framework" : framework,
                "hvf" : f_hvf,
                "molecule" : m1,
                "molecule2" : m2,
                "rosenbluth" : r1,
                "rosenbluth2" : r2
            }
    return parameters

def bench(task, j, max_iter=20):
    session_id = f"{run_id}/{benchmarking_id}/{j}"
    create_session(session_id=session_id, agent_type="RASPA", provider = "anthropic")
    session = load_session(session_id)
    agent = load_agent(session)
    agent.load(f"checkpoints/{run_id}/{i}/")
    del agent.tools["learn"]
    agent.active_learning = False
    agent.reset_chat()
    agent.auto_run = True
    response = agent.run(task, max_iter=max_iter)
    cost = agent.sum_token_count()
    save_agent(session, agent, note=task)
    
    save_session(session_id=session_id, state = session)
    # print(response)
    return response, cost


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def setup_logging(run_id: str, benchmarking_id: str, level=logging.INFO) -> logging.Logger:
    log_file = f"output/{run_id}/{benchmarking_id}/logs/run.log"
    ensure_dir(log_file)

    logger = logging.getLogger("runner")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logging initialized -> %s", log_file)
    return logger

def load_json(path: str):
    if os.path.exists(path):
        with open(path, "r") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return {}
    return {}

def write_json(path: str, data: dict, logger: logging.Logger, job_id=None):
    ensure_dir(path)
    with open(path, "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    if job_id is None:
        logger.info("Wrote JSON -> %s", path)
    else:
        logger.info("Wrote result for job %s -> %s", job_id, path)

def append_cost_jsonl(path: str, entry: dict, logger: logging.Logger, lock: threading.Lock):
    """Append a single JSON object to a .jsonl file safely."""
    ensure_dir(path)
    line = json.dumps(entry, ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    logger.info("Appended cost entry for job %s to %s", entry.get("job_id"), path)

def build_single_jobs(prompt, instruction):
    tasks_single = [
        (tasks_framework, "small"),
        (tasks_n1_s, "small"),
        (tasks_n2_ss, "small"),
        (tasks_n2_sl, "mixed"),
        (tasks_n1, "large"),
        (tasks_n2, "large"),
        (tasks_n1_l, "large"),
        (tasks_n2_ll, "large"),
    ]
    random_seeds = [x for x in range(200, 300)]
    k = len(tasks_single)  # keep original offset behavior

    jobs = []
    for task_group, mol_type in tasks_single:
        for t in task_group:
            parameters = task_parameters(
                random_seeds[k],
                small=(mol_type == "small"),
                mixed=(mol_type == "mixed"),
            )
            k += 1
            task_text = prompt + task_prompt(instruction, t, parameters)
            jobs.append({"id": k, "kind": "single", "task": task_text})
    return jobs

def build_multi_jobs(prompt, instruction):
    random_seeds = [x for x in range(300, 3000)]
    base_j = 100
    jobs = []

    for offset, t in enumerate(tasks_multistep):
        j = base_j + offset
        random.seed(random_seeds[j])
        small = random.choice([True, False])
        task_text = prompt + task_prompt(instruction, t, task_parameters(random_seeds[j], small=small))
        jobs.append({"id": j, "kind": "multi", "task": task_text})
    return jobs

def run_one(job, logger: logging.Logger):
    logger.info("Job %s started (%s)", job["id"], job["kind"])
    logger.debug("Task content for job %s: %s", job["id"], job["task"])
    res, cost = bench(job["task"], job["id"])
    logger.info("Job %s finished", job["id"])
    # Log the cost breakdown at INFO for visibility
    try:
        logger.info("Job %s cost: %s", job["id"], json.dumps(cost, ensure_ascii=False))
    except Exception:
        logger.info("Job %s cost (non-serializable): %r", job["id"], cost)
    return job["id"], job["kind"], job["task"], res, cost


def remove_cost_entries(path: str, job_ids, run_id: str, benchmarking_id: str, logger: logging.Logger):
    """Rewrite JSONL excluding entries matching (run_id, benchmarking_id, job_id in job_ids)."""
    if not os.path.exists(path):
        return
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="costs_", suffix=".jsonl", dir=os.path.dirname(path))
    os.close(tmp_fd)
    removed = 0
    with open(path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            try:
                rec = json.loads(line)
            except Exception:
                # Preserve any malformed or non-JSON lines
                dst.write(line)
                continue
            if (
                isinstance(rec, dict)
                and rec.get("run_id") == run_id
                and rec.get("benchmarking_id") == benchmarking_id
                and rec.get("job_id") in job_ids
            ):
                removed += 1
                continue
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    logger.info("Removed %d old cost entries for job_ids=%s from %s", removed, job_ids, path)


def main(max_workers=8, log_level=logging.INFO, rerun_ids=None):
    """
    rerun_ids: list[int] | None. If provided, only these job IDs are executed.
               Any previous outputs and cost entries for these IDs are removed first,
               so the rerun becomes the canonical 'first' result.
    """
    prompt = (
        "Solve problems by asking your memory for details. Find correct solutions for all subtasks! "
        "If a prerequisite is missing, the result is incorrect. IMPORTANT: ALWAYS add a readme.txt "
        "file in the end explaining the steps. Learn new insights from the simulations.\nTask: "
    )
    instruction_f1 = "Answer this question using simulations (ALWAYS USE 1/20 cycles and up to 10 molecules for speed. IGNORE the low accuracy!): "

    logger = setup_logging(run_id, benchmarking_id, level=log_level)

    single_jobs = build_single_jobs(prompt, instruction_f1)
    multi_jobs  = build_multi_jobs(prompt, instruction_f1)
    all_jobs = single_jobs + multi_jobs

    # Map id -> job for quick filtering
    jobs_by_id = {job["id"]: job for job in all_jobs}

    # Determine which jobs to run
    if rerun_ids:
        missing = [jid for jid in rerun_ids if jid not in jobs_by_id]
        if missing:
            logger.warning("Some rerun_ids not found in job list: %s", missing)
        jobs_to_run = [jobs_by_id[jid] for jid in rerun_ids if jid in jobs_by_id]
        logger.info("Rerun mode enabled for ids=%s (%d jobs filtered).", rerun_ids, len(jobs_to_run))
    else:
        jobs_to_run = all_jobs
        logger.info("Rerun mode disabled. Running all %d jobs.", len(jobs_to_run))

    output_file_single = f"output/{run_id}/{benchmarking_id}/output_single.json"
    output_file_multi  = f"output/{run_id}/{benchmarking_id}/output_multi.json"
    costs_file         = f"output/{run_id}/{benchmarking_id}/costs.jsonl"

    output_single = load_json(output_file_single)
    output_multi  = load_json(output_file_multi)

    ensure_dir(output_file_single)
    ensure_dir(output_file_multi)
    ensure_dir(costs_file)
    cost_lock = threading.Lock()

    # If rerunning, purge previous results so these become the canonical "first" run.
    if rerun_ids:
        # Figure out which rerun IDs belong to which bucket
        single_ids = [job["id"] for job in jobs_to_run if job["kind"] == "single"]
        multi_ids  = [job["id"] for job in jobs_to_run if job["kind"] == "multi"]

        # Remove from output dicts
        for jid in single_ids:
            output_single.pop(str(jid), None)  # keys become strings after json.load
            output_single.pop(jid, None)
        for jid in multi_ids:
            output_multi.pop(str(jid), None)
            output_multi.pop(jid, None)

        # Persist the cleaned files before reruns
        write_json(output_file_single, output_single, logger)
        write_json(output_file_multi,  output_multi,  logger)

        # Rewrite costs.jsonl excluding old entries for these ids
        remove_cost_entries(costs_file, set(rerun_ids), run_id, benchmarking_id, logger)

    logger.info("Prepared %d jobs (%d single, %d multi) | executing %d",
                len(all_jobs), len(single_jobs), len(multi_jobs), len(jobs_to_run))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        
        futures = {}
        for job in jobs_to_run:
            fut = ex.submit(run_one, job, logger)
            futures[fut] = job
            time.sleep(20)  # delay before submitting the next job

        for fut in as_completed(futures):
            job = futures[fut]
            try:
                id_, kind, task_text, response, cost = fut.result()

                # Store normal results (canonical)
                record = {"task": task_text, "response": response}
                if kind == "single":
                    output_single[str(id_)] = record
                    write_json(output_file_single, output_single, logger, job_id=id_)
                else:
                    output_multi[str(id_)] = record
                    write_json(output_file_multi, output_multi, logger, job_id=id_)

                # Append fresh cost entry
                cost_entry = {
                    "run_id": run_id,
                    "benchmarking_id": benchmarking_id,
                    "job_id": id_,
                    "kind": kind,
                    "cost": cost,
                }
                append_cost_jsonl(costs_file, cost_entry, logger, cost_lock)

            except Exception as e:
                logger.exception("Job %s (%s) failed: %s", job["id"], job["kind"], e)
                record = {"task": job["task"], "response": f"ERROR: {type(e).__name__}: {e}"}
                if job["kind"] == "single":
                    output_single[str(job["id"])] = record
                    write_json(output_file_single, output_single, logger, job_id=job["id"])
                else:
                    output_multi[str(job["id"])] = record
                    write_json(output_file_multi, output_multi, logger, job_id=job["id"])

                # Note an error entry in the costs file too
                error_cost_entry = {
                    "run_id": run_id,
                    "benchmarking_id": benchmarking_id,
                    "job_id": job["id"],
                    "kind": job["kind"],
                    "error": f"{type(e).__name__}: {e}",
                }
                append_cost_jsonl(costs_file, error_cost_entry, logger, cost_lock)

if __name__ == "__main__":
    # main(rerun_ids = [11, 13, 14, 15, 24, 101])
    # main(rerun_ids=[11, 15, 102])
    # main(rerun_ids=[11])
    # main(rerun_ids=[24])
    # main(rerun_ids=[10,14,16,17,20,23,101,102])# with less simulation steps and 1e4 pressure
    # main(rerun_ids=[16,20,23,102])# with less simulation steps and 1e4 pressure
    main(rerun_ids=[23])
