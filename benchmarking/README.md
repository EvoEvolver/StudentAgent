# Benchmarking

This folder contains benchmarking tools and test suites for evaluating RaspaAgent performance on RASPA simulation tasks.

## Overview

The benchmarking system provides:
- **Standardized test tasks** for RASPA simulations (helium void fraction, surface area, Henry coefficients, etc.)
- **Automated task execution** with optional hints
- **Support for single-step and multi-step simulations**
- **Progress tracking** via tqdm
- **Reproducible testing** for evaluating agent improvements

## Directory Structure

```
benchmarking/
├── scripts/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── run_feedback.py     # Feedback processing
│   ├── utils.py            # Utility functions
│   └── workflow.py         # Workflow management
├── input/
│   ├── tasks_single.json   # Single-step simulation tasks
│   ├── tasks_multi.json    # Multi-step simulation tasks
│   ├── hints.json          # Task-specific hints for the agent
│   ├── hints.py            # Hint generation utilities
│   └── tasks.py            # Task definitions
├── manual_solutions/       # Reference solutions for validation
├── output/                 # Benchmark results and outputs
└── logs/                   # Execution logs
```

## Using `run_benchmark.py`

### Basic Usage

The main entry point for benchmarking is [scripts/run_benchmark.py](scripts/run_benchmark.py). It provides a simple API for testing agent performance.

**Run from the command line:**
```bash
cd benchmarking/scripts
python run_benchmark.py
```

**Programmatic usage:**
```python
from benchmarking.scripts.run_benchmark import test_agent_n_tasks
from student.agent.agent_raspa import RaspaAgent

# Test with default settings (1 task with hints)
results = test_agent_n_tasks(n=1, give_hint=True, single=True)
```

### Key Functions

#### `test_agent_n_tasks(n, agent=None, single=True, give_hint=False)`

Test an agent on the first `n` tasks from the benchmark suite.

**Parameters:**
- `n` (int): Number of tasks to run
- `agent` (RaspaAgent, optional): Agent instance to test. If `None`, creates a new RaspaAgent at `../output/testing/`
- `single` (bool): If `True`, use single-step tasks; if `False`, use multi-step tasks
- `give_hint` (bool): If `True`, append task-specific hints to instructions

**Returns:**
- List of agent results (one per task)

**Example:**
```python
# Test 5 single-step tasks without hints
results = test_agent_n_tasks(n=5, single=True, give_hint=False)

# Test 3 multi-step tasks with hints using custom agent
my_agent = RaspaAgent(path="./my_test_output/")
results = test_agent_n_tasks(n=3, agent=my_agent, single=False, give_hint=True)
```

#### `run_task(agent, task, give_hint=False, single=True)`

Run a single task with the given agent.

**Parameters:**
- `agent` (StudentAgent): Agent to execute the task
- `task` (tuple): Task tuple of format `(task_name, task_instructions)`
- `give_hint` (bool): Whether to append hints
- `single` (bool): Whether this is a single-step task

**Returns:**
- Agent execution result

#### `run_multiple_tasks(agent, tasks, give_hint=False, single=True)`

Run multiple tasks sequentially with progress tracking.

**Parameters:**
- `agent` (StudentAgent): Agent to execute tasks
- `tasks` (list): List of task tuples
- `give_hint` (bool): Whether to append hints
- `single` (bool): Whether these are single-step tasks

**Returns:**
- List of results

### Testing New Agent Features

When developing new agent features, follow this workflow:

#### 1. **Create a Modified Agent**

```python
from student.agent.agent_raspa import RaspaAgent

# Example: Testing with a new tool
class ExperimentalRaspaAgent(RaspaAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add your experimental tool
        self.tools["my_new_tool"] = MyNewTool(path=self.path)
```

#### 2. **Run Baseline Benchmark**

First, establish baseline performance:

```python
from benchmarking.scripts.run_benchmark import test_agent_n_tasks
from student.agent.agent_raspa import RaspaAgent

# Baseline with standard agent
baseline_agent = RaspaAgent(path="./output/baseline/")
baseline_results = test_agent_n_tasks(
    n=10,
    agent=baseline_agent,
    single=True,
    give_hint=False
)
```

#### 3. **Test with Experimental Agent**

```python
# Test with modified agent
experimental_agent = ExperimentalRaspaAgent(path="./output/experimental/")
experimental_results = test_agent_n_tasks(
    n=10,
    agent=experimental_agent,
    single=True,
    give_hint=False
)
```

#### 4. **Compare Results**

```python
# Analyze results
for baseline, experimental in zip(baseline_results, experimental_results):
    print(f"Task: {baseline.get('task')}")
    print(f"Baseline success: {baseline.get('success')}")
    print(f"Experimental success: {experimental.get('success')}")
```

#### 5. **Test with Hints (Optional)**

If tasks are failing, test with hints to see if the feature improves hint utilization:

```python
results_with_hints = test_agent_n_tasks(
    n=5,
    agent=experimental_agent,
    give_hint=True
)
```

### Task Types

#### Single-Step Tasks ([tasks_single.json](input/tasks_single.json))

Simple simulations requiring one RASPA run:
- `hvf___x`: Helium void fraction calculations
- `surface___x`: Surface area determinations
- `rosenbluth___l`: Ideal gas Rosenbluth weight calculations
- `henry___s`: Henry coefficient (simple molecules)
- `henry___l`: Henry coefficient (molecules with torsions)

#### Multi-Step Tasks ([tasks_multi.json](input/tasks_multi.json))

Complex simulations requiring multiple sequential RASPA runs:
- `henry___sl`: Henry coefficient for two molecules simultaneously
- Tasks requiring helium void fraction calculation first
- Tasks requiring Rosenbluth weight calculation before Henry coefficient

### Hints System

Hints provide task-specific guidance to the agent. The [hints.json](input/hints.json) file contains:

- **Task-specific hints**: Keywords like `henry`, `hvf`, `surface`
- **Complexity hints**: `l` (molecules with torsions), `sl` (simultaneous molecules)
- **Multi-step guidance**: Instructions for sequential simulations

Hints are automatically matched based on task naming convention:
- `henry___s` → uses `henry` hint
- `henry___l` → uses `henry` + `l` hints
- `henry___sl` → uses `henry` + `l` + `sl` + `multi` hints

### Testing Strategies

**A. Feature Validation:**
Test specific capabilities (e.g., new parser, improved memory)
```python
# Test only parsing-intensive tasks
parser_tasks = [("surface___x", "Determine the surface area of IRMOF-13")]
results = run_multiple_tasks(agent, parser_tasks)
```

**B. Robustness Testing:**
Run full benchmark suite to detect regressions
```python
# Run all single-step tasks
all_results = test_agent_n_tasks(n=100, single=True)
```

**C. Learning Evaluation:**
Test multi-step reasoning improvements
```python
# Test multi-step tasks with learning enabled
learning_agent = RaspaAgent(path="./output/learning/", learning_on=True)
results = test_agent_n_tasks(n=10, agent=learning_agent, single=False)
```

**D. Hint Dependency Analysis:**
Compare performance with/without hints
```python
no_hints = test_agent_n_tasks(n=10, give_hint=False)
with_hints = test_agent_n_tasks(n=10, give_hint=True)
# Measure improvement from hints
```

## Development Tips

1. **Set `TEST = True` in [run_benchmark.py:13](scripts/run_benchmark.py#L13)** to dry-run without executing simulations
2. **Use small `n` values** during development to iterate quickly
3. **Check `output/` directory** for simulation files and results
4. **Compare against `manual_solutions/`** for validation
5. **Monitor agent conversation history** in session checkpoints

## Example: Complete Testing Workflow

```python
from benchmarking.scripts.run_benchmark import test_agent_n_tasks
from student.agent.agent_raspa import RaspaAgent

# 1. Quick smoke test (1 task)
quick_test = test_agent_n_tasks(n=1, give_hint=True)
print(f"Smoke test: {quick_test}")

# 2. Single-step benchmark (10 tasks, no hints)
single_agent = RaspaAgent(path="./output/single_benchmark/")
single_results = test_agent_n_tasks(n=10, agent=single_agent, single=True)

# 3. Multi-step benchmark (5 tasks, with hints)
multi_agent = RaspaAgent(path="./output/multi_benchmark/")
multi_results = test_agent_n_tasks(
    n=5,
    agent=multi_agent,
    single=False,
    give_hint=True
)

# 4. Analyze success rates
def success_rate(results):
    successes = sum(1 for r in results if r.get('success', False))
    return successes / len(results) if results else 0

print(f"Single-step success rate: {success_rate(single_results):.1%}")
print(f"Multi-step success rate: {success_rate(multi_results):.1%}")
```

## Related Documentation

- RaspaAgent implementation: [../src/student/agent/agent_raspa.py](../src/student/agent/agent_raspa.py)
- RASPA tools: [../src/student/agent/tools/tools_raspa.py](../src/student/agent/tools/tools_raspa.py)
