# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: MA-ROSAME (Multi-Agent Robust Safe Action Model Estimation)

MA-ROSAME is a hybrid learning framework designed to induce safe, multi-agent PDDL action models from noisy, high-dimensional execution traces. It integrates:
1. **Continuous Optimization (ROSAME Layer):** A gradient-based denoising phase that filters transient sensor anomalies.
2. **Logical Resolution (MA-SAM+ Layer):** A concurrent multi-agent SAT-based solver that maps safe actions and joint Macro-Actions.

## Environment

Python 3.12 (3.14 not yet available), venv at `.venv/`.

```powershell
# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# After cloning — initialise the ma-sam submodule
git submodule update --init --remote libs/ma-sam
```

> **Note:** `git submodule update --init` (without `--remote`) will fail because the pinned commit no longer exists on the remote. Always use `--remote`.

After initialising the submodule, re-apply two patches to `libs/ma-sam/sam_learning/core/learner_domain.py` (see **Submodule patches** below).

## Running

```powershell
# ROSAME demo (reads from problems/ by default)
python demo.py

# Multi-agent SAM learner
python run_ma_sam.py

# Multi-agent SAM+ with macro actions
python run_ma_sam_plus.py

# MA-ROSAME full pipeline (once implemented)
python run_ma_rosame.py
```

`demo.py` reads three optional env vars: `ROSAME_DATA_DIR` (default `./problems`), `ROSAME_DOMAIN_FILE` (default `blocksworld.pddl`), `ROSAME_PROBLEM_FILE` (default `0_blocksworld_prob.pddl`). Trajectory files are discovered via `*_traj.txt` glob inside `ROSAME_DATA_DIR`.

## Current Code Architecture

### ROSAME pipeline (`rosame.py` + `rosame_runner.py`)

`rosame.py` defines the PyTorch model:
- `Type`, `Predicate`, `Action_Schema` — symbolic domain primitives
- `Domain_Model` — grounds predicates/actions to propositions, runs the neural forward pass (`build()`), returns soft precondition/add-effect/delete-effect tensors
- Each `Action_Schema` has a learned `randn` embedding (raw tensor with `requires_grad=True`, shape `[n_features, 128]`) + a 5-layer MLP with 4-class softmax: irrelevant / add-effect / precondition / precondition+delete-effect
- **`Domain_Model.build(actions: list[int])`** takes a list of grounded action indices, returns three tensors of shape `[len(actions), n_propositions]`

`rosame_runner.py` provides `Rosame_Runner`:
- Constructor: `Rosame_Runner(domain_file)` — parses domain only; must then call `add_problem(problem)` to initialise `self.rosame`
- `prepare_rosame()` → converts parsed domain+problem into ROSAME's internal format
- `learn_rosame(observation, epochs)` → encodes trajectory steps as binary proposition vectors, trains via MSE loss
- `rosame_to_pddl()` / `export_rosame_domain()` → serialises learned model back to PDDL
- `export_sam_domain(learned_model, path)` → serialises a MA-SAM `LearnerDomain` to PDDL
- `check_action(action_str)` → returns the integer index for a grounded action string (handles argument-order permutations)
- `check_predicate(predicate_str)` → returns the canonical proposition string

### MA-SAM pipeline (`run_ma_sam.py`, `run_ma_sam_plus.py`)

Uses the `ma-sam` git submodule at `libs/ma-sam`. **Not pip-installable** — `ma_sam_path.py` adds it to `sys.path` at runtime. Always `import ma_sam_path` before any `sam_learning` import.

- `run_ma_sam.py` — `MultiAgentSAM.learn_combined_action_model(observations)` → `(LearnerDomain, report)`
- `run_ma_sam_plus.py` — `MASAMPlus.learn_combined_action_model_with_macro_actions(observations)` → `(LearnerDomain, report, macro_mapping)`

Both read PDDL data from `libs/ma-sam/experiments_dataset/blocks/`.

### Other files

- `cv_gridworld.py` — legacy CNN utility, unused by the main pipeline

### Submodule patches (`libs/ma-sam`)

Two bugs must be patched in `libs/ma-sam/sam_learning/core/learner_domain.py` after every submodule update:
- `_complete_missing_requirements` line 175: `.append(requirement)` → `.add(requirement)` — `requirements` is a `set` in `pddl_plus_parser>=3.17`
- `LearnerDomain.to_pddl` line 230: `action.to_pddl_legacy(...)` → `action.to_pddl()` — the legacy path passes `should_simplify` to `CompoundPrecondition.print()` which no longer accepts it

## Planned Architecture (MA-ROSAME integration)

### Development Guidelines
- **Modularity:** Do not modify `rosame.py`, `rosame_runner.py`, or anything under `libs/ma-sam/`. All integration logic lives in `ma_rosame_module/`.
- **Language:** All code comments and internal documentation must be written in English.
- **Noise injection:** Benchmarks use the random-predicate-flip model (5–20% rate).

### Pipeline Flow

```
PDDL trajectories + domain
        │
        ▼
  Rosame_Runner.learn_rosame()            ← Phase 2: train ROSAME on all steps
        │
        ▼
  noise_filter.score_steps()              ← Phase 3: MSE(predicted next state, actual next state) per step
        │  drop steps above threshold
        ▼
  adapter.to_multi_agent_observation()    ← Phase 4: re-parse with executing_agents, keep clean indices
        │
        ▼
  MASAMPlus.learn_combined_action_model_with_macro_actions(cleaned_observations)
        │
        ▼
  LearnerDomain → export PDDL
```

### Interface contracts

**`noise_filter.py`**
```python
def score_steps(rosame_runner: Rosame_Runner, observation) -> list[float]:
    """Returns MSE score per step. High score = likely noisy."""

def filter_observation(rosame_runner: Rosame_Runner, observation, threshold: float) -> tuple[any, list[int]]:
    """Returns (filtered_observation, kept_indices)."""
```

**`adapter.py`**
```python
def to_multi_agent_observation(
    domain, problem, trajectory_path: Path, agents: list[str], kept_indices: list[int]
) -> MultiAgentObservation:
    """Re-parses trajectory_path with executing_agents, keeps only steps at kept_indices."""
```

**`learner.py`**
```python
class MARosame:
    def __init__(self, domain_path, agents: list[str], noise_threshold: float = 0.1, epochs: int = 100): ...
    def fit(self, trajectory_paths: list[Path], problem_path: Path) -> tuple[LearnerDomain, dict, dict]: ...
    def export(self, learned_domain, path: Path): ...
```

### Joint Loss Function (Multi-Agent Logic)

`Domain_Model.build([idx])` returns `(precon, addeff, deleff)` tensors. For a multi-agent step with concurrent actions, compose using the **product-of-complements** rule:

```
joint_addeff[p] = 1 - Π_i (1 - addeff_i[p])
joint_deleff[p] = 1 - Π_i (1 - deleff_i[p])

predicted_s2 = s1 * (1 - joint_deleff) + (1 - s1) * joint_addeff
score        = MSE(predicted_s2, actual_s2)
```

Call `build()` with a one-element list per agent action, squeeze the batch dim, accumulate products. Use `torch.no_grad()` throughout `score_steps`.

### Noise threshold guidance

Start with `0.05–0.15`. Higher values keep more steps (less aggressive filtering). The random-predicate-flip noise model (5–20%) typically produces MSE in the `0.05–0.20` range.

---

## Implementation Plan

The table below tracks all implementation steps. Update the **Status** column as work progresses.

| # | File | What to implement | Key details | Status |
|---|------|-------------------|-------------|--------|
| 1 | `ma_rosame_module/__init__.py` | Empty package init | Just `# ma_rosame_module` | ✅ Done |
| 2 | `ma_rosame_module/noise_filter.py` | `score_steps()` | Iterate observation components; for each step call `rosame_runner.check_action()` per agent action, call `build([idx])` with `torch.no_grad()`, accumulate product-of-complements, compute MSE vs actual s2; return `list[float]` | ✅ Done |
| 3 | `ma_rosame_module/noise_filter.py` | `filter_observation()` | Call `score_steps()`, return `(observation, kept_indices)` where `kept_indices = [i for i, s in enumerate(scores) if s <= threshold]` | ✅ Done |
| 4 | `ma_rosame_module/adapter.py` | `to_multi_agent_observation()` | Re-parse `trajectory_path` via `TrajectoryParser(domain, problem).parse_trajectory(path, executing_agents=agents)`; then slice `.components` to `kept_indices` and rebuild a `MultiAgentObservation` | ✅ Done |
| 5 | `ma_rosame_module/learner.py` | `MARosame.__init__()` | Store `domain_path`, `agents`, `noise_threshold`, `epochs`; parse domain with `DomainParser` | ✅ Done |
| 6 | `ma_rosame_module/learner.py` | `MARosame.fit()` | Phase 1: parse problem + trajectories. Phase 2: `Rosame_Runner` + `learn_rosame` per trajectory. Phase 3–4: `score_steps` + `filter_observation` per trajectory. Phase 5: `adapter.to_multi_agent_observation` per trajectory. Phase 6: `MASAMPlus.learn_combined_action_model_with_macro_actions` | ✅ Done |
| 7 | `ma_rosame_module/learner.py` | `MARosame.export()` | Call `rosame_runner.export_sam_domain(learned_domain, path)` | ✅ Done |
| 8 | `run_ma_rosame.py` | CLI entry point | Mirror `run_ma_sam_plus.py`; instantiate `MARosame`, call `fit()`, print report + macro mapping, write output PDDL | ✅ Done |
| 9 | `run_ma_rosame.py` | Noise injection for testing | Before calling `fit()`, flip random predicates in trajectory copies at 10% rate to simulate noise | ✅ Done |
