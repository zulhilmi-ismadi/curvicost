import numpy as np
import pytest

import curvicost


def test_identity_is_perfect(tree2d):
    row = curvicost.score(tree2d, tree2d)
    assert row["dice"] == pytest.approx(1.0)
    assert row["iou"] == pytest.approx(1.0)
    assert row["traceable_frac"] == pytest.approx(1.0, abs=1e-9)
    assert row["conductance_frac"] == pytest.approx(1.0, abs=1e-9)
    assert row["betti0_error"] == 0


def test_identity_is_perfect_3d(tree3d):
    row = curvicost.score(tree3d, tree3d)
    assert row["dice"] == pytest.approx(1.0)
    assert row["traceable_frac"] == pytest.approx(1.0, abs=1e-9)


def test_reports_both_families(tree2d):
    row = curvicost.score(tree2d, tree2d)
    for key in ("dice", "iou", "cldice", "betti0_error", "erl", "diadem_like"):
        assert key in row, key
    for key in curvicost.COST_COLUMNS:
        assert key in row, key


def test_no_erl_skips_erl(tree2d):
    row = curvicost.score(tree2d, tree2d, with_erl=False)
    assert "erl" not in row
    assert "dice" in row and "conductance_frac" in row


def test_shape_mismatch_rejected(tree2d):
    with pytest.raises(ValueError, match="shape mismatch"):
        curvicost.score(tree2d, tree2d[:64])


def test_break_costs_more_than_it_looks(tree2d):
    """The paper's central claim, as an executable assertion.

    Cutting a branch moves few voxels -- Dice stays high -- but the branch
    beyond the cut stops being reachable, so traceable length falls much
    further than Dice does.
    """
    broken, _ = curvicost.perturb(tree2d, "break", 0.5, seed=0)
    row = curvicost.score(tree2d, broken)
    dice_loss = 1.0 - row["dice"]
    cost_loss = 1.0 - row["traceable_frac"]
    assert cost_loss > dice_loss, (
        f"expected a break to cost more function than overlap; "
        f"dice_loss={dice_loss:.4f} cost_loss={cost_loss:.4f}"
    )


def test_liberal_prediction_can_exceed_reference(tree2d):
    """traceable_frac > 1 is meaningful, not a bug: a more liberal mask can be
    more traceable than the reference (STARE's two annotators differ by 1.95)."""
    from scipy import ndimage
    fatter = ndimage.binary_dilation(tree2d, iterations=1)
    row = curvicost.score(tree2d, fatter)
    # Dice punishes the extra voxels hard; the function is almost untouched.
    dice_loss = 1.0 - row["dice"]
    trace_loss = 1.0 - row["traceable_frac"]
    assert dice_loss > 0.1, "expected dilation to move enough voxels to hurt Dice"
    assert trace_loss < dice_loss / 4, (
        f"expected function to survive dilation; "
        f"dice_loss={dice_loss:.4f} trace_loss={trace_loss:.4f}"
    )


def test_bridge_scores_well_but_merges_components(twotrees):
    """The paper's headline vignette, as an executable assertion: a spurious
    bridge is nearly invisible to Dice and obvious to topology."""
    bridged, info = curvicost.perturb(twotrees, "bridge", 0.2, seed=0)
    assert info["n_bridges"] >= 1
    row = curvicost.score(twotrees, bridged)
    assert row["dice"] > 0.98, "a bridge should barely move any voxels"
    assert row["betti0_error"] >= 1, "a bridge should merge components"
