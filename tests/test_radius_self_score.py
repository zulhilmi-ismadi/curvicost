"""The radius arm's reference scores 1 on every metric against itself.

The audit scores the radius operator against its own scale-1.0 reconstruction. This pins
the contract at the metric level: prepare the reference the way `sweep()` prepares the
radius arm's, score it against itself, and require the ideal value from every metric --
including the DIADEM-like score, whose reference graph must be built from that same mask.
"""
import numpy as np
import curvicost
from curvicost.score import prepare_reference, score


def test_radius_reference_scores_ideal_against_itself(tree2d):
    ref, _ = curvicost.perturb(tree2d, "radius", 1.0)
    ctx = prepare_reference(ref, with_erl=True)
    row = score(ref, ref, with_erl=True, reference=ctx)
    for k in ("dice", "iou", "cldice", "erl_frac", "diadem_like", "traceable_frac", "conductance_frac"):
        assert abs(row[k] - 1.0) < 1e-9, f"{k} = {row[k]} for the reference against itself"
    assert row["betti0_error"] == 0
