import torch
import torch.nn.functional as F


def score_steps(rosame_runner, observation) -> list[float]:
    """Returns MSE score per step. High score = likely noisy."""
    scores = []
    n_props = len(rosame_runner.rosame.propositions)

    with torch.no_grad():
        for component in observation.components:
            joint_action = component.grounded_joint_action
            operational_actions = joint_action.operational_actions

            # Encode previous and next states as binary proposition vectors
            pre_state = {
                rosame_runner.check_predicate(pred.untyped_representation[1:-1])
                for _, val in component.previous_state.state_predicates.items()
                for pred in val
            }
            next_state = {
                rosame_runner.check_predicate(pred.untyped_representation[1:-1])
                for _, val in component.next_state.state_predicates.items()
                for pred in val
            }

            s1 = torch.tensor(
                [1.0 if p in pre_state else 0.0 for p in rosame_runner.rosame.propositions]
            )
            s2 = torch.tensor(
                [1.0 if p in next_state else 0.0 for p in rosame_runner.rosame.propositions]
            )

            # Accumulate product-of-complements over all agent actions in this step
            complement_add = torch.ones(n_props)
            complement_del = torch.ones(n_props)

            valid = False
            for action in operational_actions:
                action_str = action.__str__()[1:-1]
                idx = rosame_runner.check_action(action_str)
                if idx is None:
                    continue
                valid = True
                _, addeff, deleff = rosame_runner.rosame.build([idx])
                complement_add *= (1.0 - addeff[0])
                complement_del *= (1.0 - deleff[0])

            if not valid:
                scores.append(0.0)
                continue

            joint_add = 1.0 - complement_add
            joint_del = 1.0 - complement_del

            predicted_s2 = s1 * (1.0 - joint_del) + (1.0 - s1) * joint_add
            score = F.mse_loss(predicted_s2, s2).item()
            scores.append(score)

    return scores


def filter_observation(rosame_runner, observation, threshold: float):
    """Returns (observation, kept_indices). kept_indices are the clean step positions."""
    scores = score_steps(rosame_runner, observation)
    kept_indices = [i for i, s in enumerate(scores) if s <= threshold]
    return observation, kept_indices
