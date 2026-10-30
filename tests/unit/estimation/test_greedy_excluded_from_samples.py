"""The greedy decode is not a sample: only the `greedy` method may read it."""
from src.estimation.estimation_pipeline import _compute_arm_estimate
from src.estimation.estimation_structure import TrajectoryScoringData


def traj(idx, scores, greedy=False):
    return TrajectoryScoringData(traj_idx=idx, arm="trunk", structure_scores=scores, conditional_logprobs={},
                                 n_generated_tokens=10, text="", is_greedy=greedy)


def test_sample_statistics_ignore_the_greedy_decode():
    samples = [traj(0, [0.0, 1.0]), traj(1, [0.0, 1.0]), traj(2, [1.0, 0.0]), traj(3, [1.0, 1.0])]
    est = _compute_arm_estimate(0, "trunk", samples + [traj(-1, [1.0, 0.0], greedy=True)], {"root": {}, "trunk": {}})
    assert est.estimates["uniform"].core == [0.5, 0.75]
    assert est.estimates["greedy"].core == [1.0, 0.0]
    assert sorted(t.traj_idx for t in est.trajectories) == [0, 1, 2, 3]


def test_spread_and_order_statistics_ignore_the_greedy_decode():
    samples = [traj(0, [0.0, 1.0]), traj(1, [0.0, 1.0]), traj(2, [0.0, 1.0]), traj(3, [1.0, 0.0])]
    refs = {"root": {}, "trunk": {}}
    alone = _compute_arm_estimate(0, "trunk", samples, refs)
    mixed = _compute_arm_estimate(0, "trunk", samples + [traj(-1, [1.0, 0.0], greedy=True)], refs)
    for method in ("uniform", "median", "mode", "prob"):
        assert mixed.estimates[method].core == alone.estimates[method].core
        assert mixed.estimates[method].deviance_avg == alone.estimates[method].deviance_avg
