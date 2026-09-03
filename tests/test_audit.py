import json

import numpy as np
import pytest

import curvicost
from curvicost.audit import sweep, audit, DEFAULT_SEVERITIES
from curvicost.cli import main

# 3+ severities per operator: below that audit() cannot correlate and skips.
FAST = {"break": (0.1, 0.3, 0.5), "bridge": (0.1, 0.3, 0.5),
        "truncate": (0.1, 0.3, 0.5), "radius": (0.7, 0.85, 1.15, 1.3),
        "boundary": (0.02, 0.1, 0.2)}


@pytest.fixture(scope="module")
def rows(twotrees):
    return sweep(twotrees, severities=FAST, unit="twotrees")


def test_sweep_covers_every_operator(rows):
    got = {r["operator"] for r in rows}
    assert got == set(curvicost.OPERATORS), f"missing: {set(curvicost.OPERATORS) - got}"


def test_sweep_rows_carry_metrics_and_costs(rows):
    r = rows[0]
    for key in ("dice", "cldice", "traceable_frac", "conductance_frac",
                "operator", "severity", "seed", "unit"):
        assert key in r, key


def test_radius_is_scored_against_its_own_reconstruction(tree2d):
    """Scale 1.0 must be the radius baseline. Scored against the raw mask the
    rasterisation loss is charged to the operator and Dice never reaches 1."""
    out = sweep(tree2d, severities={"radius": (1.15,)}, unit="t")
    assert out, "radius produced no cases"
    # A mild thickening scored against its own reconstruction stays close to 1.
    assert out[0]["dice"] > 0.6


def test_audit_returns_findings(rows):
    res = audit(rows, n_boot=50)
    assert res["n_cases"] == len(rows)
    assert res["findings"]
    for f in res["findings"]:
        assert set(f) >= {"cost", "operator", "metric", "aligned_rho", "blind", "anti"}


def test_single_unit_reports_no_ci(rows):
    """One mask cannot give an honest cluster CI, and the result must say so
    rather than inventing one."""
    res = audit(rows, n_boot=50)
    assert res["n_units"] == 1
    assert res["has_ci"] is False
    assert all(np.isnan(f["ci_lo"]) for f in res["findings"])


def test_multiple_units_do_get_cis(tree2d, twotrees):
    rows = []
    for i, m in enumerate((tree2d, twotrees, tree2d[::-1], twotrees[::-1])):
        rows += sweep(m, severities={"break": (0.1, 0.3, 0.5)}, unit=f"u{i}")
    res = audit(rows, n_boot=100)
    assert res["n_units"] == 4 and res["has_ci"] is True
    assert any(np.isfinite(f["ci_lo"]) for f in res["findings"])


def test_betti0_is_sign_aligned(twotrees):
    """betti0_error counts errors, so its raw correlation runs backwards. The
    audit must report it aligned or it reads as disagreeing when it agrees."""
    rows = sweep(twotrees, severities={"break": (0.1, 0.3, 0.5, 0.7)}, unit="t")
    res = audit(rows, n_boot=50)
    f = next(f for f in res["findings"]
             if f["metric"] == "betti0_error" and f["cost"] == "traceable_frac")
    d = next(f for f in res["findings"]
             if f["metric"] == "dice" and f["cost"] == "traceable_frac")
    if np.isfinite(f["aligned_rho"]) and np.isfinite(d["aligned_rho"]):
        assert np.sign(f["aligned_rho"]) == np.sign(d["aligned_rho"]), (
            "aligned Betti-0 should agree in sign with Dice on the same data")


def test_empty_rows_is_not_a_crash():
    res = audit([], n_boot=10)
    assert res["n_cases"] == 0 and res["findings"] == []


def test_too_few_severities_is_reported_not_silent(twotrees):
    """Two severities cannot be correlated. The result must say which operators
    it dropped, or the user sees an empty table and no reason for it."""
    rows = sweep(twotrees, severities={"break": (0.1, 0.5)}, unit="t")
    res = audit(rows, n_boot=20)
    assert res["findings"] == []
    assert "break" in res["skipped"] and res["skipped"]["break"] == 2


def test_cli_explains_an_empty_report(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    main(["audit", str(p), "--severities", "0.1,0.5", "--n-boot", "20"])
    out = capsys.readouterr().out
    assert "skipped for too few cases" in out


def test_cli_audit_writes_json_and_csv(tmp_path, twotrees):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    j, c = tmp_path / "a.json", tmp_path / "a.csv"
    rc = main(["audit", str(p), "--json", str(j), "--csv", str(c),
               "--severities", "0.1,0.3,0.5", "--n-boot", "50", "--quiet"])
    assert rc == 0
    res = json.loads(j.read_text())
    assert res["n_cases"] > 0 and res["findings"]
    assert c.read_text().splitlines()[0].startswith("cost,operator,metric")


def test_cli_audit_warns_when_ci_impossible(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    main(["audit", str(p), "--severities", "0.1,0.3,0.5", "--n-boot", "20"])
    out = capsys.readouterr().out
    assert "WITHOUT" in out and "confidence" in out


def test_scale_report_measures_prune_in_vessel_radii(tree2d):
    from curvicost.audit import scale_report
    r = scale_report(tree2d, prune_px=5)
    assert r["median_radius"] > 0
    assert r["prune_in_radii"] == pytest.approx(5 / r["median_radius"])


def test_scale_report_tracks_thickness(tree2d):
    """A thicker structure must report a LOWER prune-in-radii for the same
    prune_px -- that is the whole point of the diagnostic."""
    from scipy import ndimage
    from curvicost.audit import scale_report
    thick = ndimage.binary_dilation(tree2d, iterations=3)
    assert scale_report(thick, 5)["prune_in_radii"] < \
           scale_report(tree2d, 5)["prune_in_radii"]


def test_cli_audit_warns_on_mismatched_scale(tmp_path, tree2d, capsys):
    """prune_px is an absolute length; on a very thick structure it prunes far
    less than the ~2 radii the tool was calibrated on, and must say so."""
    from scipy import ndimage
    from curvicost.io import save_mask
    thick = ndimage.binary_dilation(tree2d, iterations=6)
    p = save_mask(thick, tmp_path / "thick.npy")
    main(["audit", str(p), "--severities", "0.1,0.3,0.5", "--n-boot", "20",
          "--prune-px", "1"])
    out = capsys.readouterr().out
    assert "scale:" in out and "vessel radii" in out
    assert "not\n     comparable across datasets" in out or "comparable across" in out
