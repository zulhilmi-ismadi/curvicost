import numpy as np
import pytest

import curvicost
from curvicost.perturb import OPERATORS


@pytest.mark.parametrize("operator", ["break", "bridge", "truncate", "boundary"])
def test_operator_changes_the_mask(twotrees, operator):
    out, info = curvicost.perturb(twotrees, operator, 0.2, seed=0)
    assert out.shape == twotrees.shape
    assert out.dtype == bool
    assert not np.array_equal(out, twotrees), f"{operator} was a no-op"


def test_bridge_needs_a_close_pair(tree2d):
    """A bridge requires two tips near in space but far in the graph. A clean
    single tree offers no such pair, and the operator declines rather than
    inventing an invalid connection."""
    out, info = curvicost.perturb(tree2d, "bridge", 0.2, seed=0)
    assert info["n_bridges"] == 0
    assert np.array_equal(out, tree2d)


def test_radius_scales_thickness(tree2d):
    thin, _ = curvicost.perturb(tree2d, "radius", 0.7)
    fat, _ = curvicost.perturb(tree2d, "radius", 1.3)
    assert thin.sum() < fat.sum()


def test_radius_baseline_is_its_own_reconstruction(tree2d):
    """Documented contract: scale 1.0 is the reference for the radius operator."""
    ref, _ = curvicost.perturb(tree2d, "radius", 1.0)
    assert ref.sum() > 0
    row = curvicost.score(ref, ref)
    assert row["dice"] == pytest.approx(1.0)


def test_severity_is_monotone_in_damage(tree2d):
    light = curvicost.score(tree2d, curvicost.perturb(tree2d, "boundary", 0.01, seed=0)[0])
    heavy = curvicost.score(tree2d, curvicost.perturb(tree2d, "boundary", 0.20, seed=0)[0])
    assert heavy["dice"] < light["dice"]


def test_unknown_operator_rejected(tree2d):
    with pytest.raises(ValueError, match="unknown operator"):
        curvicost.perturb(tree2d, "smudge", 0.1)


def test_empty_mask_rejected():
    with pytest.raises(ValueError, match="empty"):
        curvicost.perturb(np.zeros((32, 32), bool), "break", 0.1)


def test_info_reports_realised_severity(tree2d):
    _, info = curvicost.perturb(tree2d, "break", 0.5, seed=0)
    assert "requested_frac" in info and "n_applied" in info
    assert info["n_applied"] >= 1
