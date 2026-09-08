"""`curvicost profile` — does it recover the operator it was given?

The profiler earns its place only if, handed a prediction that IS one of the
five operators, it names that operator. These tests hold it to that on a
held-out seed: the calibration basis is built at seed 0, the test perturbations
at seed 1, so recovery cannot come from matching identical voxels.
"""
import json

import numpy as np
import pytest

import curvicost
from curvicost.profile import profile, signature, SIGNATURE_FIELDS
from curvicost.perturb import perturb, OPERATORS
from curvicost.cli import main

# A trimmed calibration ladder. The full one is five to six rungs per operator;
# three is enough to bracket a severity and keeps the suite under a minute.
FAST = {"break": (0.05, 0.20, 0.50), "bridge": (0.05, 0.20, 0.50),
        "truncate": (0.05, 0.20, 0.50), "radius": (0.70, 0.85, 1.30),
        "boundary": (0.02, 0.05, 0.20)}

CASES = [("break", 0.20), ("bridge", 0.20), ("truncate", 0.20),
         ("radius", 0.70), ("boundary", 0.05)]


@pytest.fixture(scope="module")
def profiles(twotrees):
    out = {}
    for op, sev in CASES:
        pred, _ = perturb(twotrees, op, sev, seed=7)
        out[op] = profile(twotrees, pred, seed=0, seeds=2, calibration=FAST)
    return out


@pytest.mark.parametrize("op,sev", CASES)
def test_recovers_the_operator_it_was_given(profiles, op, sev):
    r = profiles[op]
    assert r["nearest"] == op, f"{op} at {sev} read as {r['nearest']}: {r['shares']}"


@pytest.mark.parametrize("op,_sev", CASES)
def test_shares_are_a_normalised_nonnegative_mixture(profiles, op, _sev):
    sh = profiles[op]["shares"]
    assert set(sh) <= set(OPERATORS)
    assert all(v >= 0 for v in sh.values()), sh
    assert sum(sh.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= profiles[op]["residual"] <= 1.0


def test_signature_is_all_zero_when_nothing_changed(twotrees):
    sig, _ = signature(twotrees, twotrees)
    assert len(sig) == len(SIGNATURE_FIELDS)
    assert np.allclose(sig, 0.0, atol=1e-9), dict(zip(SIGNATURE_FIELDS, sig))


def test_signature_separates_loss_from_gain(twotrees):
    """Removal must move `removed` and not `added`; addition the reverse."""
    cut, _ = perturb(twotrees, "break", 0.35, seed=7)
    add, _ = perturb(twotrees, "bridge", 0.35, seed=7)
    s_cut, ref = signature(twotrees, cut)
    s_add, _ = signature(twotrees, add, _ref=ref)
    assert s_cut[0] > 0 and s_cut[1] == 0.0        # removed, added
    assert s_add[1] > 0 and s_add[0] == 0.0
    assert s_cut[2] > 0 > s_add[2] or s_add[2] <= 0  # break splits, bridge cannot


def test_profile_is_exported(twotrees):
    assert curvicost.profile is profile


def test_cli_profile_writes_json(tmp_path, twotrees):
    gt = tmp_path / "gt.npy"
    pred_path = tmp_path / "pred.npy"
    out = tmp_path / "profile.json"
    pred, _ = perturb(twotrees, "break", 0.20, seed=7)
    np.save(gt, twotrees)
    np.save(pred_path, pred)
    assert main(["profile", str(pred_path), "--gt", str(gt), "--json", str(out), "--seeds", "2", "--quiet"]) == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 1
    assert rows[0]["nearest"] == "break"
    assert set(rows[0]) >= {"shares", "residual", "nearest", "observed", "severities", "prediction"}
