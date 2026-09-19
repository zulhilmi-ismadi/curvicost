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
    for key in ("dice", "cldice", "traceable_frac", "conductance_frac", "conductance_twosided",
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
    res = audit(rows, n_boot=20, attempted=["break"])
    assert res["findings"] == []
    assert "break" in res["notes"] and "3+" in res["notes"]["break"]


def test_cli_explains_an_empty_report(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    main(["audit", str(p), "--severities", "0.1,0.5", "--n-boot", "20"])
    out = capsys.readouterr().out
    assert "needs 3+ to correlate" in out


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


def test_declined_operator_is_explained(tree2d):
    """`bridge` legitimately declines on a clean single tree. The result must
    say why rather than leaving a blank column."""
    rows = sweep(tree2d, severities={"bridge": (0.1, 0.3, 0.5)}, unit="t")
    res = audit(rows, n_boot=20, attempted=["bridge"])
    assert rows == [] or all(r["operator"] != "bridge" for r in rows)
    assert "bridge" in res["notes"] and "declined" in res["notes"]["bridge"]


def test_cost_that_does_not_move_is_undefined_not_blind(twotrees):
    """Reachable length counts reference voxels, so a bridge between two
    reference fragments cannot change it. The audit must report those cells as
    undefined -- a statement about the cost -- and not as metric blindness."""
    rows = sweep(twotrees, severities={"bridge": (0.1, 0.3, 0.5)}, unit="t")
    assert len(rows) >= 3, "fixture no longer admits three bridge cases"
    res = audit(rows, n_boot=20, attempted=["bridge"])
    cells = [f for f in res["findings"] if f["cost"] == "traceable_frac"]
    assert cells and all(f["undefined"] for f in cells)
    assert not any(f["blind"] for f in cells)
    assert "did not move" in res["notes"]["bridge"]


def test_cli_renders_undefined_cells(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    main(["audit", str(p), "--severities", "0.1,0.3,0.5", "--n-boot", "20"])
    out = capsys.readouterr().out
    assert "undef" in out


def test_few_units_is_inconclusive_whatever_the_interval(tree2d, twotrees):
    """The paper's floor: a cell in which the operator acted in fewer than 10
    units is inconclusive REGARDLESS of its interval -- a narrow interval on
    four masks is not a validated correlation either. Never blind, never anti."""
    rows = []
    for i, m in enumerate((tree2d, twotrees, tree2d[::-1], twotrees[::-1])):
        rows += sweep(m, severities={"break": (0.1, 0.3, 0.5)}, unit=f"u{i}")
    res = audit(rows, n_boot=200)
    assert res["n_units"] == 4 and res["min_units"] == 10 and res["min_cases"] == 12
    defined = [f for f in res["findings"] if not f["undefined"]]
    assert defined
    for f in defined:
        assert f["label"] == "inconclusive" and f["reason"].startswith("<10 units")
        assert f["inconclusive"] and not f["blind"] and not f["anti"]
        assert f["n_units"] == 4 and f["n"] == len([r for r in rows if r["operator"] == "break"])
    assert "break" in res["notes"] and "INCONCLUSIVE" in res["notes"]["break"]


# ---- the paper's four labels, and the per-cell floors -----------------------------

from curvicost.audit import label_cell, MIN_UNITS_FOR_BLINDNESS, MIN_CASES


def test_floors_are_the_papers():
    assert MIN_UNITS_FOR_BLINDNESS == 10 and MIN_CASES == 12


@pytest.mark.parametrize("rho,lo,hi,n_cases,n_units,expect,reason", [
    (np.nan, np.nan, np.nan, 100, 20, "undefined", ""),          # constant cost or metric
    (0.9, 0.8, 0.95, 100, 9, "inconclusive", "<10 units"),      # narrow interval, too few units
    (0.9, 0.8, 0.95, 11, 20, "inconclusive", "<12 cases"),      # too few cases
    (0.9, 0.8, 0.95, 11, 9, "inconclusive", "<10 units, <12 cases"),
    (-0.9, -0.95, -0.8, 100, 9, "inconclusive", "<10 units"),   # negative but below floor: NOT anti
    (0.1, -0.2, 0.4, 100, 20, "blind", ""),                     # interval includes zero
    (-0.1, -0.4, 0.2, 100, 20, "blind", ""),                    # negative point estimate, includes zero
    (-0.5, -0.7, -0.3, 100, 20, "anti-correlated", ""),        # negative, excludes zero
    (-0.05, -0.09, -0.01, 100, 20, "anti-correlated", ""),      # no -0.2 threshold any more
    (0.5, 0.3, 0.7, 100, 20, "tracks", ""),
    (0.5, 0.3, 0.7, 12, 10, "tracks", ""),                      # exactly at both floors
])
def test_label_definitions(rho, lo, hi, n_cases, n_units, expect, reason):
    assert label_cell(rho, lo, hi, n_cases, n_units) == (expect, reason)


def test_label_precedence_undefined_beats_floor():
    assert label_cell(np.nan, np.nan, np.nan, 3, 1)[0] == "undefined"


def test_unit_floor_is_counted_per_operator_cell(tree2d, twotrees):
    """An operator that acts on few of the masks has fewer acting units than
    the audit has masks; the floor must be counted per cell, not once."""
    from curvicost.audit import METRIC_ORDER
    # Twelve distinct "masks", but `bridge` only ever acts on the twotrees copies.
    rows = []
    masks = [tree2d, twotrees, tree2d[::-1], twotrees[::-1], tree2d.T, twotrees.T,
             tree2d[:, ::-1], twotrees[:, ::-1], tree2d[::-1, ::-1], twotrees[::-1, ::-1],
             tree2d.T[::-1], twotrees.T[::-1]]
    for i, m in enumerate(masks):
        rows += sweep(m, severities={"break": (0.1, 0.3, 0.5), "bridge": (0.1, 0.3, 0.5)},
                      unit=f"u{i}")
    res = audit(rows, n_boot=50, attempted=["break", "bridge"])
    assert res["n_units"] == 12
    brk = [f for f in res["findings"] if f["operator"] == "break"]
    brg = [f for f in res["findings"] if f["operator"] == "bridge"]
    assert brk and all(f["n_units"] == 12 for f in brk)
    assert all(f["label"] != "inconclusive" for f in brk if not f["undefined"]), \
        "12 acting units and 36 cases clear both floors"
    assert brg, "bridge produced no cells on the twotrees copies"
    assert all(f["n_units"] == 6 for f in brg), "bridge acts on the six twotrees copies only"
    assert all(f["label"] in ("inconclusive", "undefined") for f in brg)
    assert all(f["reason"].startswith("<10 units") for f in brg if f["label"] == "inconclusive")


def test_case_floor_is_counted_per_operator_cell(tree2d, twotrees):
    """Ten units but one severity each: 10 acting cases < 12 -> inconclusive (<12 cases)."""
    rows = []
    masks = [tree2d, twotrees, tree2d[::-1], twotrees[::-1], tree2d.T, twotrees.T,
             tree2d[:, ::-1], twotrees[:, ::-1], tree2d[::-1, ::-1], twotrees[::-1, ::-1]]
    for i, m in enumerate(masks):
        rows += sweep(m, severities={"break": (0.3,)}, unit=f"u{i}")
    res = audit(rows, n_boot=50, attempted=["break"])
    cells = [f for f in res["findings"] if not f["undefined"]]
    assert cells
    assert all(f["n_units"] == 10 and f["n"] == 10 for f in cells)
    assert all(f["label"] == "inconclusive" and f["reason"] == "<12 cases" for f in cells)


def test_findings_carry_label_interval_and_counts(rows):
    res = audit(rows, n_boot=50)
    for f in res["findings"]:
        assert f["label"] in ("undefined", "inconclusive", "blind", "anti-correlated", "tracks")
        assert {"n", "n_units", "ci_lo", "ci_hi", "reason"} <= set(f)
        assert f["undefined"] == (f["label"] == "undefined")
        assert f["inconclusive"] == (f["label"] == "inconclusive")
        assert f["blind"] == (f["label"] == "blind")
        assert f["anti"] == (f["label"] == "anti-correlated")


def test_cli_prints_every_cell_with_interval_and_counts(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    main(["audit", str(p), "--severities", "0.1,0.3,0.5", "--n-boot", "20"])
    out = capsys.readouterr().out
    assert "every cell" in out and "cases" in out and "units" in out
    assert "inconclusive (<10 units" in out
    assert "anti-correlated" in out and "undefined" in out and "blind" in out


def test_cli_csv_has_label_and_counts(tmp_path, twotrees):
    from curvicost.io import save_mask
    import csv as _csv
    p = save_mask(twotrees, tmp_path / "m.npy")
    c = tmp_path / "a.csv"
    main(["audit", str(p), "--csv", str(c), "--severities", "0.1,0.3,0.5",
          "--n-boot", "20", "--quiet"])
    hdr = c.read_text().splitlines()[0].split(",")
    for col in ("label", "reason", "n", "n_units", "ci_lo", "ci_hi", "blind", "anti",
                "inconclusive", "undefined", "cost_constant"):
        assert col in hdr, col
    rows_ = list(_csv.DictReader(c.open()))
    assert all(r["label"] in ("undefined", "inconclusive") for r in rows_), \
        "one mask can never produce a verdict"


def test_cli_ladder_options(tmp_path, twotrees):
    """--radius-scales / --boundary-fracs / --study-ladder must reach sweep()."""
    from curvicost.io import save_mask
    from curvicost.audit import STUDY_SEVERITIES
    p = save_mask(twotrees, tmp_path / "m.npy")
    j = tmp_path / "a.json"
    main(["audit", str(p), "--json", str(j), "--quiet", "--n-boot", "5",
          "--severities", "0.1,0.5", "--radius-scales", "0.85,1.15",
          "--boundary-fracs", "0.01,0.02"])
    lad = json.loads(j.read_text())["ladder"]
    assert lad["severities"]["radius"] == [0.85, 1.15]
    assert lad["severities"]["boundary"] == [0.01, 0.02]
    assert lad["severities"]["break"] == [0.1, 0.5] and lad["seeds"] == [0] and lad["n_boot"] == 5
    # --study-ladder sets the study's values; explicit options still override.
    parser_args = __import__("curvicost.cli", fromlist=["build_parser"]).build_parser().parse_args(
        ["audit", str(p), "--study-ladder"])
    assert parser_args.study_ladder and parser_args.seeds is None
    assert STUDY_SEVERITIES["break"] == (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
    assert STUDY_SEVERITIES["radius"] == (0.50, 0.70, 0.85, 1.15, 1.30)
    assert STUDY_SEVERITIES["boundary"] == (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


# --------------------------------------------------------------------------
# 0.2.1: user-supplied quantities, density_kept, redraw floor
# --------------------------------------------------------------------------

def _foreground_count(mask):
    """A toy user quantity: foreground voxel count."""
    return float(np.count_nonzero(mask))


def test_mask_quantity_sidedness():
    from curvicost.audit import MaskQuantity
    one = MaskQuantity("q1", _foreground_count, "one")
    two = MaskQuantity("q2", _foreground_count, "two")
    kept = MaskQuantity("q3", _foreground_count, "kept")
    assert one.read(100.0, 150.0) == pytest.approx(1.5)
    assert two.read(100.0, 150.0) == pytest.approx(1 / 1.5)
    assert two.read(100.0, 50.0) == pytest.approx(0.5)
    assert two.read(100.0, 0.0) == 0.0
    assert kept.read(100.0, 150.0) == pytest.approx(0.5)
    assert kept.read(100.0, 80.0) == pytest.approx(0.8)
    assert np.isnan(one.read(0.0, 5.0)), "a zero reference has no fraction"
    with pytest.raises(ValueError):
        MaskQuantity("bad", _foreground_count, "three")


def test_density_kept_is_the_studys_definition(twotrees):
    """1 - |d_pred/d_ref - 1|, d the foreground count (mask_quantities.py), with the
    radius arm read against its own redrawn reference."""
    from curvicost.audit import MASK_QUANTITIES
    q = MASK_QUANTITIES["density_kept"]
    rows = sweep(twotrees, severities=FAST, unit="t", quantities=[q])
    radius_ref, _ = curvicost.perturb(twotrees, "radius", 1.0)
    for r in rows:
        base = radius_ref if r["operator"] == "radius" else twotrees
        out, _ = curvicost.perturb(twotrees, r["operator"], r["severity"], seed=r["seed"])
        expect = 1.0 - abs(out.sum() / base.sum() - 1.0)
        assert r["density_kept"] == pytest.approx(expect, abs=1e-12), r["operator"]


def test_quantities_leave_existing_columns_untouched(twotrees, rows):
    from curvicost.audit import MASK_QUANTITIES
    with_q = sweep(twotrees, severities=FAST, unit="twotrees",
                   quantities=[MASK_QUANTITIES["density_kept"]])
    assert len(with_q) == len(rows)
    for a, b in zip(rows, with_q):
        assert set(b) - set(a) == {"density_kept"}
        for k in a:
            assert json.dumps(a[k]) == json.dumps(b[k]), k
    base = audit(rows, n_boot=30)
    both = audit(with_q, n_boot=30, costs=("traceable_frac", "conductance_twosided", "density_kept"))
    old = [f for f in both["findings"] if f["cost"] != "density_kept"]
    assert json.dumps(base["findings"]) == json.dumps(old)


@pytest.mark.parametrize("sided", ["one", "two"])
def test_cli_cost_fn_both_sidedness(tmp_path, twotrees, sided, monkeypatch, capsys):
    from curvicost.io import save_mask
    mod = tmp_path / "toy_quantities.py"
    mod.write_text("import numpy as np\n"
                   "def area(mask):\n    return float(np.count_nonzero(mask))\n")
    monkeypatch.chdir(tmp_path)
    p = save_mask(twotrees, tmp_path / "m.npy")
    j = tmp_path / "a.json"
    rc = main(["audit", str(p), "--json", str(j), "--include-cases", "--n-boot", "5",
               "--severities", "0.1,0.3,0.5", "--cost-fn", "toy_quantities:area",
               "--cost-sided", sided])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vs user-supplied toy_quantities:area" in out
    assert ("one-sided" if sided == "one" else "two-sided") in out
    res = json.loads(j.read_text())
    assert res["cost_sided"] == {"user:area": sided}
    assert any(f["cost"] == "user:area" for f in res["findings"])
    # built-in defaults are still audited beside the user quantity
    assert {f["cost"] for f in res["findings"]} == {"traceable_frac", "conductance_twosided",
                                                    "user:area"}
    radius_ref, _ = curvicost.perturb(twotrees, "radius", 1.0)
    vals = []
    for r in res["cases"]:
        base = radius_ref if r["operator"] == "radius" else twotrees
        out_m, _ = curvicost.perturb(twotrees, r["operator"], r["severity"], seed=r["seed"])
        c = out_m.sum() / base.sum()
        expect = c if sided == "one" else min(c, 1 / c)
        assert r["user:area"] == pytest.approx(expect, abs=1e-12)
        vals.append(r["user:area"])
    if sided == "one":
        assert max(vals) > 1.0, "a thickening must read above 1 one-sided"
    else:
        assert max(vals) <= 1.0


def test_cli_cost_fn_accepts_a_file_path(tmp_path, twotrees):
    from curvicost.io import save_mask
    mod = tmp_path / "q.py"
    mod.write_text("def half(mask):\n    return mask.sum() / 2.0\n")
    p = save_mask(twotrees, tmp_path / "m.npy")
    c = tmp_path / "a.csv"
    assert main(["audit", str(p), "--csv", str(c), "--quiet", "--n-boot", "5",
                 "--severities", "0.1,0.3,0.5", "--cost-fn", f"{mod}:half"]) == 0
    assert "user:half" in c.read_text()


def test_cli_cost_fn_bad_spec_is_a_clean_error(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    assert main(["audit", str(p), "--quiet", "--cost-fn", "no_colon_here"]) == 2
    assert main(["audit", str(p), "--quiet", "--cost-fn", "curvicost_no_such_mod:f"]) == 2
    assert main(["audit", str(p), "--quiet", "--cost-fn", "numpy:no_such_function"]) == 2
    assert "--cost-fn" in capsys.readouterr().err


def test_redraw_floor_scores_the_redrawn_reference(twotrees):
    from curvicost.audit import redraw_floor, MASK_QUANTITIES
    from curvicost.score import score
    q = MASK_QUANTITIES["density_kept"]
    row = redraw_floor(twotrees, unit="t", quantities=[q])
    redrawn, _ = curvicost.perturb(twotrees, "radius", 1.0)
    direct = score(twotrees, redrawn)
    for k in ("dice", "cldice", "erl_frac", "traceable_frac", "conductance_twosided"):
        assert row[k] == pytest.approx(direct[k], abs=1e-12), k
    assert row["traceable_frac"] == pytest.approx(1.0), "the redraw keeps the centre line"
    assert row["density_kept"] == pytest.approx(1 - abs(redrawn.sum() / twotrees.sum() - 1))
    assert row["operator"] == "redraw" and row["unit"] == "t"


def test_audit_attaches_redraw_floor(tree2d, twotrees):
    from curvicost.audit import redraw_floor
    rows, floors = [], []
    for i, m in enumerate((tree2d, twotrees, tree2d[::-1])):
        rows += sweep(m, severities={"break": (0.1, 0.3, 0.5)}, unit=f"u{i}")
        floors.append(redraw_floor(m, unit=f"u{i}"))
    plain = audit(rows, n_boot=30)
    res = audit(rows, n_boot=30, floors=floors)
    fl = res["redraw_floor"]
    assert fl["n_units"] == 3
    d = [f["conductance_twosided"] for f in floors]
    assert fl["values"]["conductance_twosided"]["median"] == pytest.approx(np.median(d))
    assert fl["values"]["conductance_twosided"]["min"] == pytest.approx(min(d))
    for f, g in zip(res["findings"], plain["findings"]):
        assert f["redraw_metric"] == pytest.approx(fl["values"][f["metric"]]["median"])
        assert f["redraw_cost"] == pytest.approx(fl["values"][f["cost"]]["median"])
        assert json.dumps({k: v for k, v in f.items()
                           if k not in ("redraw_metric", "redraw_cost")}) == json.dumps(g)


def test_cli_prints_redraw_floor_and_writes_it(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    import csv as _csv
    p = save_mask(twotrees, tmp_path / "m.npy")
    c = tmp_path / "a.csv"
    main(["audit", str(p), "--csv", str(c), "--severities", "0.1,0.3,0.5", "--n-boot", "5",
          "--cost", "traceable_frac", "--cost", "density_kept"])
    out = capsys.readouterr().out
    assert "redraw floor of traceable_frac: 1.000" in out
    assert "redraw floor of density_kept:" in out
    assert "redraw" in out.split("vs traceable length")[1].splitlines()[1]
    # the quantity heads each table and the report opens by naming it
    assert "scored against:" in out.split("vs ")[0]
    assert "vs traceable length (reachable reference skeleton)" in out
    assert "vs vessel density kept" in out
    rows_ = list(_csv.DictReader(c.open()))
    assert {"redraw_metric", "redraw_cost"} <= set(rows_[0])
    tr = [r for r in rows_ if r["cost"] == "traceable_frac"]
    assert all(float(r["redraw_cost"]) == pytest.approx(1.0) for r in tr)


def test_cli_no_redraw_floor_keeps_the_020_columns(tmp_path, twotrees, capsys):
    from curvicost.io import save_mask
    p = save_mask(twotrees, tmp_path / "m.npy")
    c = tmp_path / "a.csv"
    main(["audit", str(p), "--csv", str(c), "--severities", "0.1,0.3,0.5", "--n-boot", "5",
          "--no-redraw-floor"])
    assert "redraw" not in capsys.readouterr().out
    assert c.read_text().splitlines()[0] == ("cost,operator,metric,n,n_units,aligned_rho,ci_lo,"
                                             "ci_hi,label,reason,undefined,inconclusive,blind,"
                                             "anti,cost_constant")
