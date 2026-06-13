import ma_sam_path  # noqa: F401 – adds libs/ma-sam to sys.path
from collections import defaultdict
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Observation
from sam_learning.learners import MASAMPlus

from rosame_runner import Rosame_Runner
from ma_rosame_module.noise_filter import filter_observation
from ma_rosame_module.adapter import to_multi_agent_observation

_NOP_NAMES = {"nop", "dummy-add-predicate-action", "dummy-del-predicate-action"}


def _to_single_agent_observation(ma_obs) -> Observation:
    """Extract a pseudo-single-agent Observation from a MultiAgentObservation.

    Per step: pick the first operational action whose name is not a NOP.
    Used only for ROSAME training — ROSAME expects single-action steps.
    """
    sa_obs = Observation()
    sa_obs.add_problem_objects(ma_obs.grounded_objects)
    for component in ma_obs.components:
        for action in component.grounded_joint_action.operational_actions:
            if action.name.lower() not in _NOP_NAMES:
                sa_obs.add_component(
                    previous_state=component.previous_state,
                    call=action,
                    next_state=component.next_state,
                )
                break
    return sa_obs


def _union_objects(problems) -> dict:
    """Merge objects from all problems into one type→[names] dict (deduped)."""
    merged = defaultdict(set)
    for problem in problems:
        for name, obj in problem.objects.items():
            merged[obj.type.name].add(name)
    return {t: sorted(names) for t, names in merged.items()}


class MARosame:
    def __init__(self, domain_path, agents: list[str], noise_threshold: float = 0.1, epochs: int = 100):
        self.domain_path = Path(domain_path)
        self.agents = agents
        self.noise_threshold = noise_threshold
        self.epochs = epochs
        self.domain = DomainParser(self.domain_path, partial_parsing=True).parse_domain()

    def fit(self, trajectory_paths: list[Path], problem_paths: list[Path]):
        """Run the full MA-ROSAME pipeline.

        Each trajectory_path must have a corresponding problem_path at the same index.
        Returns (learned_domain, report, macro_mapping).
        """
        trajectory_paths = [Path(p) for p in trajectory_paths]
        problem_paths = [Path(p) for p in problem_paths]

        # Phase 1 — parse each (problem, trajectory) pair as multi-agent observations
        pairs = []
        for traj_path, prob_path in zip(trajectory_paths, problem_paths):
            problem = ProblemParser(problem_path=prob_path, domain=self.domain).parse_problem()
            ma_obs = TrajectoryParser(self.domain, problem).parse_trajectory(
                traj_path, executing_agents=self.agents
            )
            pairs.append((traj_path, problem, ma_obs))

        # Phase 2 — joint ROSAME training across all trajectories
        #
        # Ground the model once to the union of all objects so every trajectory
        # is encoded against the same proposition space. This lets us concatenate
        # all steps into one dataset and train with a single optimizer pass,
        # rather than independently fitting each trajectory in sequence.
        rosame_runner = Rosame_Runner(self.domain_path)
        problems = [problem for _, problem, _ in pairs]
        union_objects = _union_objects(problems)

        # add_problem needs a problem object for the rosame init path;
        # use the first one, then immediately re-ground to the union
        rosame_runner.add_problem(problems[0])
        rosame_runner.rosame.ground_from_dict(union_objects)
        rosame_runner.objects = union_objects

        # Collect pseudo-single-agent steps from all trajectories into one observation
        combined_sa_obs = Observation()
        combined_sa_obs.add_problem_objects(
            {name: obj for problem in problems for name, obj in problem.objects.items()}
        )
        for _, _, ma_obs in pairs:
            sa_obs = _to_single_agent_observation(ma_obs)
            for component in sa_obs.components:
                combined_sa_obs.components.append(component)

        print(f"Joint training on {len(combined_sa_obs.components)} steps from {len(pairs)} trajectories")
        rosame_runner.learn_rosame(combined_sa_obs, epochs=self.epochs)

        # Phases 3–5 — score, filter, and rebuild each trajectory
        # Re-ground per-problem for scoring so propositions align with the
        # actual objects present in each trajectory's states.
        cleaned_observations = []
        for traj_path, problem, ma_obs in pairs:
            rosame_runner.add_problem(problem)
            rosame_runner.ground_new_trajectory()
            _, kept_indices = filter_observation(rosame_runner, ma_obs, self.noise_threshold)
            print(f"{traj_path.name}: {len(kept_indices)}/{len(ma_obs.components)} steps kept")
            ma_obs_clean = to_multi_agent_observation(
                self.domain, problem, traj_path, self.agents, kept_indices
            )
            cleaned_observations.append(ma_obs_clean)

        # Phase 6 — symbolic learning with MA-SAM+
        learner = MASAMPlus(self.domain)
        learned_domain, report, macro_mapping = (
            learner.learn_combined_action_model_with_macro_actions(cleaned_observations)
        )
        return learned_domain, report, macro_mapping

    def export(self, learned_domain, path: Path):
        """Write learned_domain to a PDDL file."""
        Path(path).write_text(learned_domain.to_pddl())
