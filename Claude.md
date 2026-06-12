# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: MA-ROSAME (Multi-Agent Robust Safe Action Model Estimation)

MA-ROSAME is a hybrid learning framework designed to induce safe, multi-agent PDDL action models from noisy, high-dimensional execution traces. It integrates:
1. **Continuous Optimization (ROSAME Layer):** A gradient-based denoising phase that filters transient sensor anomalies.
2. **Logical Resolution (MA-SAM+ Layer):** A concurrent multi-agent SAT-based solver that maps safe actions and joint Macro-Actions.

## Environment

Python 3.14, venv at `.venv/`.

```powershell
# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# After cloning — initialise the ma-sam submodule
git submodule update --init
```

## Running

```powershell
# ROSAME demo (reads from problems/ by default)
python demo.py

# Multi-agent SAM learner
python run_ma_sam.py

# Multi-agent SAM+ with macro actions
python run_ma_sam_plus.py
```

`demo.py` reads three optional env vars: `ROSAME_DATA_DIR` (default `./problems`), `ROSAME_DOMAIN_FILE` (default `blocksworld.pddl`), `ROSAME_PROBLEM_FILE` (default `0_blocksworld_prob.pddl`). Trajectory files are discovered via `*_traj.txt` glob inside `ROSAME_DATA_DIR`.

## Current Code Architecture

### ROSAME pipeline (`rosame.py` + `rosame_runner.py`)

`rosame.py` defines the PyTorch model:
- `Type`, `Predicate`, `Action_Schema` — symbolic domain primitives
- `Domain_Model` — grounds predicates/actions to propositions, runs the neural forward pass (`build()`), returns soft precondition/add-effect/delete-effect tensors
- Each `Action_Schema` has a learned `randn` embedding + a 5-layer MLP with 4-class softmax: irrelevant / add-effect / precondition / precondition+delete-effect

`rosame_runner.py` provides `Rosame_Runner`, bridging `pddl_plus_parser` PDDL parsing with the model:
- `prepare_rosame()` → converts parsed domain+problem into ROSAME's internal format
- `learn_rosame(observation, epochs)` → encodes trajectory steps as binary proposition vectors, trains via MSE loss
- `rosame_to_pddl()` / `export_rosame_domain()` → serialises learned model back to PDDL
- `export_sam_domain(learned_model, path)` → serialises a MA-SAM `LearnerDomain` to PDDL

### MA-SAM pipeline (`run_ma_sam.py`, `run_ma_sam_plus.py`)

Uses the `ma-sam` git submodule at `libs/ma-sam`. **Not pip-installable** — `ma_sam_path.py` adds it to `sys.path` at runtime. Always `import ma_sam_path` before any `sam_learning` import.

- `run_ma_sam.py` — `MultiAgentSAM.learn_combined_action_model(observations)` → `(LearnerDomain, report)`
- `run_ma_sam_plus.py` — `MASAMPlus.learn_combined_action_model_with_macro_actions(observations)` → `(LearnerDomain, report, macro_mapping)`

Both read PDDL data from `libs/ma-sam/experiments_dataset/blocks/`.

### Submodule patches (`libs/ma-sam`)

Two bugs were fixed in `libs/ma-sam/sam_learning/core/learner_domain.py` to work with `pddl_plus_parser>=3.17` (the version this repo requires). Do not revert these:
- `_complete_missing_requirements`: `.append()` → `.add()` — `requirements` is a `set` in newer parser versions
- `LearnerDomain.to_pddl`: calls `action.to_pddl()` instead of `action.to_pddl_legacy()` — the legacy path passed `should_simplify` to `CompoundPrecondition.print()` which no longer accepts it

## Planned Architecture (MA-ROSAME integration)

### Core Architecture
- **Stage 1: Denoising Phase:** Operates on `MultiAgentObservation` sequences. Uses continuous weight tensors to minimize soft-loss and filter out non-consistent transitions.
- **Stage 2: Logical Resolution Phase:** Consumes sanitized observations to build a CNF matrix. Resolves multi-agent ambiguities using the MA-SAM+ structural binder.

### Development Guidelines
- **Consistency:** Always prioritize the consistency of logical predicates after the denoising phase.
- **Language:** All code comments and internal documentation must be written in English.
- **Modularity:** Maintain strict separation: do not modify existing `ma-sam` or `rosame_runner` codebases. Implement all integration logic within a new `ma_rosame_module`.
- **Integration:** The project relies on `pddl_plus_parser` for domain modeling and the `ma-sam` logical engine.
- **Experimentation:** Noise injection in benchmarks should follow the random-predicate-flip model (typically 5%-20% rate).

### Key Components (to be built)

All integration code lives in `ma_rosame_module/`. Do not modify `rosame.py`, `rosame_runner.py`, or anything under `libs/ma-sam/`.

- `ma_rosame_module/learner.py` — orchestrates the full pipeline (entry point)
- `ma_rosame_module/noise_filter.py` — scores trajectory steps with ROSAME and drops noisy ones
- `ma_rosame_module/adapter.py` — converts filtered single-agent observations to `MultiAgentObservation`

### Pipeline Flow

```
PDDL trajectories + domain
        │
        ▼
  Rosame_Runner.learn_rosame()        ← train ROSAME on all steps
        │
        ▼
  noise_filter.score_steps()          ← MSE(ROSAME predicted state, actual next state) per step
        │  drop steps above threshold
        ▼
  adapter.to_multi_agent_observation() ← re-parse trajectory files with executing_agents,
        │                                  keep only clean step indices
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
    """Returns (filtered_observation, kept_indices). kept_indices is used by the adapter."""
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

### Noise threshold guidance

The threshold for `filter_observation` is the MSE between ROSAME's predicted next-state vector and the actual observed next-state vector (both are binary proposition vectors). Start with `0.05–0.15`; higher values keep more steps (less aggressive filtering). The random-predicate-flip noise model (5–20% rate) typically produces MSE in the `0.05–0.20` range.

### Joint Loss Function (Multi-Agent Logic)

ROSAME's `build()` returns `(precon, addeff, deleff)` tensors for a single action. For a multi-agent step with concurrent actions `{a_1, ..., a_n}`, call `build()` once per agent action and compose the results using the **product-of-complements** rule (the probability that *at least one* agent produces the effect):

```
joint_addeff[p] = 1 - Π_i (1 - addeff_i[p])
joint_deleff[p] = 1 - Π_i (1 - deleff_i[p])
```

This is mathematically correct for independent concurrent effects — unlike clamped sum, it stays in `[0, 1]` and handles overlapping agent effects cleanly.

The per-step noise score is then:

```
predicted_s2 = state_1 * (1 - joint_deleff) + (1 - state_1) * joint_addeff
score        = MSE(predicted_s2, actual_s2)   # averaged over propositions
```

Steps where `score > threshold` are flagged as noisy and dropped before passing to MA-SAM+.

**Implementation note for `score_steps`:** call `rosame_runner.rosame.build(agent_action_index)` once per agent in the step (not once per step), accumulate the product, then compute MSE. Use `torch.no_grad()` — this is inference only, no backprop.

### MA-ROSAME Algorithm (Pseudocode)

```
ALGORITHM MA-ROSAME
INPUT:
    domain_path          — PDDL domain file
    trajectory_paths     — list of multi-agent trajectory files
    problem_path         — PDDL problem file
    agents               — list of agent names ["a1", "a2", ...]
    noise_threshold τ    — MSE cutoff (e.g. 0.1)
    epochs               — ROSAME training epochs

OUTPUT:
    learned_domain       — PDDL action model
    report               — safe/unsafe action summary
    macro_mapping        — macro action bindings

═══════════════════════════════════════════════════════════════
PHASE 1 — PARSE
═══════════════════════════════════════════════════════════════

domain  ← DomainParser(domain_path)
problem ← ProblemParser(problem_path, domain)

for each traj in trajectory_paths:
    MA_obs[traj] ← TrajectoryParser.parse(traj, executing_agents=agents)
    // returns MultiAgentObservation with joint steps

═══════════════════════════════════════════════════════════════
PHASE 2 — TRAIN ROSAME  (single-agent surrogate)
═══════════════════════════════════════════════════════════════

rosame ← Rosame_Runner(domain_path, problem)

for each traj in trajectory_paths:
    // Extract pseudo-single-agent steps
    for each step in MA_obs[traj]:
        a_first ← first non-NOP action in step.joint_actions
        SA_steps[traj] ← (step.s1, a_first, step.s2)

    rosame.learn(SA_steps[traj], epochs)
    // trains MLP per action schema via:
    // loss = MSE(s1*(1-del) + (1-s1)*add, s2)
    //      + MSE((1-s1)*pre, 0)
    //      + 0.2 * MSE(pre, 1)

═══════════════════════════════════════════════════════════════
PHASE 3 — SCORE STEPS  (joint multi-agent loss)  ← NEW
═══════════════════════════════════════════════════════════════

for each traj in trajectory_paths:
    for each step k in MA_obs[traj]:
        s1 ← encode(step.previous_state)   // binary proposition vector
        s2 ← encode(step.next_state)

        // Initialise product accumulators
        complement_add ← ones(n_propositions)
        complement_del ← ones(n_propositions)

        for each agent action aᵢ in step.operational_actions:
            _, addeff_i, deleff_i ← rosame.build(aᵢ)  // no backprop
            complement_add ← complement_add ⊙ (1 - addeff_i)
            complement_del ← complement_del ⊙ (1 - deleff_i)

        // Product-of-complements: P(at least one agent causes effect)
        joint_add ← 1 - complement_add
        joint_del ← 1 - complement_del

        // Predict next state under joint action
        predicted_s2 ← s1 ⊙ (1 - joint_del) + (1 - s1) ⊙ joint_add

        score[traj][k] ← MSE(predicted_s2, s2)

═══════════════════════════════════════════════════════════════
PHASE 4 — FILTER NOISY STEPS
═══════════════════════════════════════════════════════════════

for each traj in trajectory_paths:
    clean_steps[traj] ← { k : score[traj][k] ≤ τ }
    cleaned_obs[traj] ← MA_obs[traj].keep(clean_steps[traj])

═══════════════════════════════════════════════════════════════
PHASE 5 — SYMBOLIC LEARNING  (MA-SAM+)
═══════════════════════════════════════════════════════════════

learner ← MASAMPlus(domain)
learned_domain, report, macro_mapping ←
    learner.learn_combined_action_model_with_macro_actions(cleaned_obs)

═══════════════════════════════════════════════════════════════
PHASE 6 — EXPORT
═══════════════════════════════════════════════════════════════

write learned_domain.to_pddl() → output file

return learned_domain, report, macro_mapping
```

The novel contribution is Phase 3 — the joint multi-agent loss via product-of-complements. Phases 2 and 5 reuse existing algorithms unchanged.
