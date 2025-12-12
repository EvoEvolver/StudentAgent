import concurrent.futures
import pickle
from datetime import datetime

from run_benchmark import run_task_i
from tqdm import tqdm

RUN_NAME = f"run_0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
agent_params = dict(
    provider="anthropic",
    path=f"../output/{RUN_NAME}/",
    expensive=True,
    verbose=False,
    memory_path="../memory/initial_knowledge",
)


def run_single_task(i):
    print(f"Starting task {i}")
    r, agent = run_task_i(
        agent_params=agent_params, i=i, single=True, give_hint=True, variant_index=0
    )
    return i, r, agent


if __name__ == "__main__":
    results = {}
    tasks = [9]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_single_task, i) for i in tasks]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                i, r, agent = future.result()
                results[i] = (r, agent)
                print(f"Finished task {i}")
            except Exception as e:
                print(f"Task generated an exception: {e}")

    pickle.dump(results, open(f"../output/{RUN_NAME}/results.pkl", "wb"))
