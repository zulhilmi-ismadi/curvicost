import numpy as np
import pytest

import curvicost


def test_identity_is_perfect(tree2d):
    row = curvicost.score(tree2d, tree2d)
    assert row["dice"] == pytest.approx(1.0)
    assert row["iou"] == pytest.approx(1.0)
    assert row["traceable_frac"] == pytest.approx(1.0, abs=1e-9)
    assert row["conductance_frac"] == pytest.approx(1.0, abs=1e-9)
    assert row["conductance_twosided"] == pytest.approx(1.0, abs=1e-9)
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


def test_dilation_keeps_reachable_length(tree2d):
    """A more liberal mask keeps every reference voxel reachable: reachable
    length stays at 1.0 while Dice falls. (Under the superseded graph-based
    definition this read as traceable_frac > 1; the reach cost counts reference
    voxels and is bounded by 1 by construction.)"""
    from scipy import ndimage
    fatter = ndimage.binary_dilation(tree2d, iterations=1)
    row = curvicost.score(tree2d, fatter)
    assert 1.0 - row["dice"] > 0.1, "expected dilation to move enough voxels to hurt Dice"
    assert row["traceable_frac"] == pytest.approx(1.0, abs=1e-9)
    assert row["traceable_single_frac"] == pytest.approx(1.0, abs=1e-9)


def test_reachable_length_is_bounded_by_one(twotrees):
    """No prediction can be more reachable than the reference under the
    multi-root cost of record. The single-source variant is NOT bounded: a
    bridge between two reference fragments lets one source reach the other
    fragment, and on a fragmented reference that reads as a gain -- which is
    precisely why the study traces every reference component from its own root."""
    from scipy import ndimage
    preds = [ndimage.binary_dilation(twotrees, iterations=2),
             curvicost.perturb(twotrees, "bridge", 0.5, seed=0)[0],
             curvicost.perturb(twotrees, "boundary", 0.1, seed=0)[0]]
    for pred in preds:
        row = curvicost.score(twotrees, pred)
        assert row["traceable_frac"] <= 1.0 + 1e-12
    bridged = curvicost.score(twotrees, preds[1])
    assert bridged["traceable_frac"] == pytest.approx(1.0, abs=1e-12)
    assert bridged["traceable_single_frac"] > 1.0, "twotrees has two components; a bridge joins them"


def test_break_is_verified_to_sever(tree2d):
    """Every cut the break operator reports must actually disconnect the mask,
    and the engine must say when one could not (info["unsevered"])."""
    broken, info = curvicost.perturb(tree2d, "break", 0.5, seed=0)
    assert info["n_broken"] >= 1
    assert info["unsevered"] == 0
    row = curvicost.score(tree2d, broken)
    assert row["betti0_error"] >= 1
    assert row["traceable_frac"] < 1.0


def test_bridge_does_not_read_as_a_flow_loss(twotrees):
    """Kirchhoff conductance is a resistor-network quantity: adding a path can
    never lower it (Rayleigh monotonicity). Under the superseded path-sum a
    bridge read as a 41% conductance loss on the study data."""
    bridged, info = curvicost.perturb(twotrees, "bridge", 0.2, seed=0)
    assert info["n_bridges"] >= 1
    row = curvicost.score(twotrees, bridged)
    assert row["conductance_frac"] >= 1.0 - 1e-6, row["conductance_frac"]


def test_two_sided_conductance_charges_excess(tree2d):
    """A thicker prediction conducts more than the reference (raw fraction > 1);
    the reported cost treats that excess as loss, symmetrically in log space."""
    from scipy import ndimage
    fatter = ndimage.binary_dilation(tree2d, iterations=2)
    row = curvicost.score(tree2d, fatter)
    assert row["conductance_frac"] > 1.0
    assert row["conductance_twosided"] == pytest.approx(1.0 / row["conductance_frac"], rel=1e-9)
    assert row["conductance_twosided"] < 1.0


def test_root_loss_does_not_zero_the_cost(tree2d):
    """Truncating the branch that holds a component's root must not count the whole
    component as unreachable: the component is traced from its nearest surviving
    reference-skeleton voxel instead (the root fallback)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from curvicost._core.reach import ref_roots_for
    skel = skeletonize(tree2d)
    ref_lab, roots = ref_roots_for(tree2d, skel, ndimage.distance_transform_edt(tree2d))
    (root,) = roots.values()
    pred = tree2d.copy()
    r0, c0 = root
    pred[max(r0 - 6, 0):r0 + 7, max(c0 - 6, 0):c0 + 7] = False     # punch out the root itself
    row = curvicost.score(tree2d, pred)
    assert row["traceable_frac"] > 0.3, row["traceable_frac"]


def test_bridge_scores_well_but_merges_components(twotrees):
    """The paper's headline vignette, as an executable assertion: a spurious
    bridge is nearly invisible to Dice and obvious to topology."""
    bridged, info = curvicost.perturb(twotrees, "bridge", 0.2, seed=0)
    assert info["n_bridges"] >= 1
    row = curvicost.score(twotrees, bridged)
    assert row["dice"] > 0.98, "a bridge should barely move any voxels"
    assert row["betti0_error"] >= 1, "a bridge should merge components"
