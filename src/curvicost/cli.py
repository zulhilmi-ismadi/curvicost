"""Command line interface.

    curvicost score seg.tif --gt gt.tif
    curvicost perturb gt.tif --operator break --severity 0.1 -o broken.tif
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from . import __version__
from .io import load_mask, save_mask, voxel_size_of

_METRIC_ORDER = ("dice", "iou", "cldice", "betti0_error", "erl_frac", "diadem_like")
_COST_ORDER = ("traceable_frac", "traceable_single_frac", "conductance_twosided",
               "conductance_frac", "perfused_of_self")


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _report(row, stream):
    """Human-readable output: metrics beside costs, which is the whole point."""
    print("metric                 value   |  cost                      value", file=stream)
    print("-" * 68, file=stream)
    left = [(k, row[k]) for k in _METRIC_ORDER if k in row]
    right = [(k, row[k]) for k in _COST_ORDER if k in row]
    for i in range(max(len(left), len(right))):
        lk, lv = left[i] if i < len(left) else ("", "")
        rk, rv = right[i] if i < len(right) else ("", "")
        lcell = f"{lk:<18} {_fmt(lv):>8}" if lk else " " * 27
        rcell = f"{rk:<22} {_fmt(rv):>8}" if rk else ""
        print(f"{lcell}   |  {rcell}".rstrip(), file=stream)


def _cmd_score(args):
    gt, gt_vox = load_mask(args.gt)
    pred, _ = load_mask(args.prediction)
    from .score import score

    row = score(gt, pred, with_erl=not args.no_erl, prune_px=args.prune_px)
    row = {"prediction": str(args.prediction), "reference": str(args.gt), **row}

    vox = voxel_size_of(args.voxel_size, gt.ndim) or gt_vox
    if vox is not None:
        row["voxel_size"] = ",".join(str(v) for v in vox)
        if len(set(vox)) > 1 and not args.quiet:
            print(
                f"note: anisotropic voxels {vox} -- lengths and radii are reported in "
                "VOXELS, not physical units. See README 'Voxel size'.",
                file=sys.stderr,
            )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(row, fh, indent=2)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    if not args.quiet:
        if args.json or args.csv:
            _report(row, sys.stdout)
        else:
            json.dump(row, sys.stdout, indent=2)
            print()
    return 0


_COST_LABEL = {"traceable_frac": "traceable length (reachable reference skeleton)",
               "conductance_twosided": "conductance (Kirchhoff, two-sided: excess is loss)",
               "conductance_frac": "conductance (Kirchhoff, raw fraction retained)"}


_LABEL_MARK = {"blind": "*", "anti-correlated": "!", "inconclusive": "?", "tracks": " "}


def _audit_report(result, stream):
    """The blindness report: a grid per cost (metrics down, operators across) and
    then every cell in full -- rho, its interval, acting cases and acting units --
    with the paper's label."""
    from .audit import OPERATORS, METRIC_ORDER

    findings = result["findings"]
    metrics = [m for m in METRIC_ORDER if any(f["metric"] == m for f in findings)]
    min_units = result.get("min_units", 10)
    min_cases = result.get("min_cases", 12)
    print(f"\n{result['n_cases']} cases from {result['n_units']} reference mask(s)",
          file=stream)
    if not result["has_ci"]:
        print("⚠  fewer than 3 reference masks: correlations are reported WITHOUT "
              "confidence\n   intervals, and blindness cannot be judged. Treat these as "
              "indicative only.", file=stream)

    def cell(cost, op, m):
        return next((f for f in findings if f["cost"] == cost and f["operator"] == op
                     and f["metric"] == m), None)

    costs = list(dict.fromkeys(f["cost"] for f in findings))
    for cost in costs:
        print(f"\n  vs {_COST_LABEL.get(cost, cost)}", file=stream)
        print("  " + "metric".ljust(14) + "".join(o.rjust(11) for o in OPERATORS),
              file=stream)
        for m in metrics:
            cells = ""
            for op in OPERATORS:
                f = cell(cost, op, m)
                if f is None:
                    cells += "         --"
                elif f["label"] == "undefined":
                    cells += "      undef"
                elif f["label"] == "inconclusive":
                    # Printed for the record, in parentheses: NOT a validated
                    # correlation, never a verdict.
                    cells += f"{'(' + format(f['aligned_rho'], '+.2f') + ')':>10}?"
                else:
                    cells += f"{f['aligned_rho']:>+10.2f}{_LABEL_MARK[f['label']]}"
            print("  " + m.ljust(14) + cells, file=stream)

    # Every cell in full. The grid above is the summary; this is the evidence.
    print("\n  every cell: sign-aligned Spearman ρ, 95% cluster-bootstrap interval, acting "
          "cases / acting units, label", file=stream)
    for cost in costs:
        print(f"\n  vs {_COST_LABEL.get(cost, cost)}", file=stream)
        print("  " + "operator".ljust(10) + "metric".ljust(14) + "ρ".rjust(7)
              + "  [   lo,    hi]" + "  cases" + "  units" + "  label", file=stream)
        for op in OPERATORS:
            for m in metrics:
                f = cell(cost, op, m)
                if f is None:
                    continue
                rho = f["aligned_rho"]
                if f["label"] == "undefined":
                    num, ci = "  undef", "  [    --,    --]"
                else:
                    num = f"{rho:>+7.3f}"
                    lo, hi = f["ci_lo"], f["ci_hi"]
                    ci = ("  [" + (f"{lo:+.2f}" if lo == lo else "   --").rjust(6) + ", "
                          + (f"{hi:+.2f}" if hi == hi else "   --").rjust(6) + "]")
                lab = f["label"] + (f" ({f['reason']})" if f.get("reason") else "")
                print("  " + op.ljust(10) + m.ljust(14) + num + ci
                      + f"{f['n']:>7}{f['n_units']:>7}  {lab}", file=stream)

    sc = result.get("scale")
    if sc:
        print(f"\n  scale: median vessel radius {sc['median_radius']:.1f} px; "
              f"--prune-px {sc['prune_px']} = {sc['prune_in_radii']:.1f} vessel radii.",
              file=stream)
        # The study cells run ~2.0-2.5 radii. FIVES at 0.75 demonstrably
        # shifted several correlations, so the band has to exclude it.
        if not (1.0 <= sc["prune_in_radii"] <= 4.0):
            print("  ⚠  that is far from the ~2 radii this tool was calibrated on. "
                  "Spur pruning is an\n     ABSOLUTE pixel length, so results are not "
                  "comparable across datasets sampled at\n     different resolutions "
                  "unless you match this ratio.", file=stream)
    for op, why in sorted(result.get("notes", {}).items()):
        print(f"\n⚠  {op}: {why}", file=stream)
    if not findings:
        print("\nNo correlations could be computed. See above.", file=stream)
        return
    count = lambda lab: sum(1 for f in findings if f["label"] == lab)
    n = len(findings)
    print(f"\n  labels ({n} cells; definitions as in the paper's Methods):", file=stream)
    print(f"  undef  undefined — the cost or the metric is constant across the cell, so ρ "
          f"does not exist   ({count('undefined')})", file=stream)
    print(f"  (ρ)?   inconclusive — the operator acted in fewer than {min_units} units or "
          f"fewer than {min_cases} cases; ρ is shown\n         for the record but is NOT a "
          f"validated correlation and is never read as blind or anti-correlated   "
          f"({count('inconclusive')})", file=stream)
    print(f"  *      blind — the interval includes zero: the metric cannot see that error "
          f"type on this data   ({count('blind')})", file=stream)
    print(f"  !      anti-correlated — ρ is negative and the interval excludes zero: the "
          f"metric moves the WRONG WAY   ({count('anti-correlated')})", file=stream)
    print(f"         tracks — ρ is positive and the interval excludes zero   "
          f"({count('tracks')})", file=stream)
    anti = [f for f in findings if f["label"] == "anti-correlated"]
    if anti:
        worst = min(anti, key=lambda f: f["aligned_rho"])
        print(f"    worst: {worst['metric']} vs {_COST_LABEL.get(worst['cost'], worst['cost'])} on "
              f"{worst['operator']} (ρ {worst['aligned_rho']:+.2f}). When the metric "
              f"moves the wrong way on an\n    error type, check first whether the "
              "cost is the right one for that error type.", file=stream)


def _parse_floats(text):
    return tuple(float(v) for v in text.split(",") if v.strip())


def _cmd_audit(args):
    import json as _json

    import numpy as np
    from .audit import (sweep, audit, scale_report, DEFAULT_SEVERITIES, STUDY_SEVERITIES,
                        STUDY_SEEDS, STUDY_N_BOOT, COSTS, MIN_UNITS_FOR_BLINDNESS, MIN_CASES)

    if args.study_ladder:
        severities = dict(STUDY_SEVERITIES)
        n_seeds = len(STUDY_SEEDS) if args.seeds is None else args.seeds
        n_boot = STUDY_N_BOOT if args.n_boot is None else args.n_boot
    else:
        severities = dict(DEFAULT_SEVERITIES)
        n_seeds = 1 if args.seeds is None else args.seeds
        n_boot = 1000 if args.n_boot is None else args.n_boot
    if args.severities:
        vals = _parse_floats(args.severities)
        for op in ("break", "bridge", "truncate"):
            severities[op] = vals
    if args.radius_scales:
        severities["radius"] = _parse_floats(args.radius_scales)
    if args.boundary_fracs:
        severities["boundary"] = _parse_floats(args.boundary_fracs)

    rows, seeds, scales = [], tuple(range(n_seeds)), []
    for path in args.masks:
        mask, _ = load_mask(path)
        if not args.quiet:
            print(f"sweeping {path} ...", file=sys.stderr, flush=True)
        scales.append(scale_report(mask, args.prune_px))
        rows += sweep(mask, severities=severities, seeds=seeds,
                      prune_px=args.prune_px, with_erl=not args.no_erl,
                      unit=str(path))
    attempted = [op for op in severities if severities.get(op)]
    kw = dict(costs=tuple(args.cost) if args.cost else COSTS,
              min_units=args.min_units if args.min_units is not None else MIN_UNITS_FOR_BLINDNESS,
              min_cases=args.min_cases if args.min_cases is not None else MIN_CASES)
    result = audit(rows, n_boot=n_boot, attempted=attempted, **kw)
    result["ladder"] = dict(severities={k: list(v) for k, v in severities.items()},
                            seeds=list(seeds), n_boot=n_boot,
                            prune_px=args.prune_px, version=__version__)
    if scales:
        radii = [s["median_radius"] for s in scales]
        ratios = [s["prune_in_radii"] for s in scales]
        result["scale"] = dict(median_radius=float(np.median(radii)),
                               prune_px=args.prune_px,
                               prune_in_radii=float(np.median(ratios)))

    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(dict(result, cases=rows if args.include_cases else None),
                       fh, indent=2, default=float)
    if args.csv and result["findings"]:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(result["findings"][0]))
            w.writeheader(); w.writerows(result["findings"])
    if not args.quiet:
        _audit_report(result, sys.stdout)
    return 0


def _cmd_profile(args):
    import json as _json
    import numpy as np
    from .profile import profile

    gt, _ = load_mask(args.gt)
    rows = []
    for path in args.predictions:
        pred, _ = load_mask(path)
        if pred.shape != gt.shape:
            print(f"curvicost: {path} has shape {pred.shape}, reference has {gt.shape}; skipped",
                  file=sys.stderr)
            continue
        if not args.quiet:
            print(f"profiling {path} ...", file=sys.stderr, flush=True)
        r = profile(gt, pred, prune_px=args.prune_px, seed=args.seed, seeds=args.seeds)
        r["prediction"] = str(path)
        rows.append(r)
    if not rows:
        print("curvicost: nothing to profile", file=sys.stderr)
        return 2
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(rows, fh, indent=2, default=float)
    if not args.quiet:
        _profile_report(rows, sys.stdout)
    return 0


def _profile_report(rows, stream):
    """Which error types does this method make, and how much is outside the model?"""
    from .audit import OPERATORS
    print(f"\n{len(rows)} prediction(s) of ONE method against one reference\n", file=stream)
    print("  " + "prediction".ljust(26) + "".join(o.rjust(10) for o in OPERATORS)
          + "  unexplained", file=stream)
    flagged = False
    for r in rows:
        name = str(r["prediction"]).rsplit("/", 1)[-1]
        cells = "".join(f"{100 * r['shares'].get(o, 0.0):>9.0f}%" for o in OPERATORS)
        mark = " +" if r["residual"] > 0.30 else "  "
        flagged |= r["residual"] > 0.30
        print("  " + name[:26].ljust(26) + cells + f"{100 * r['residual']:>10.0f}%{mark}", file=stream)
    if len(rows) > 1:
        mean = {o: sum(r["shares"].get(o, 0.0) for r in rows) / len(rows) for o in OPERATORS}
        resid = sum(r["residual"] for r in rows) / len(rows)
        print("  " + "MEAN".ljust(26) + "".join(f"{100 * mean[o]:>9.0f}%" for o in OPERATORS)
              + f"{100 * resid:>10.0f}%", file=stream)
        lead = max(mean, key=mean.get)
    else:
        lead = rows[0]["nearest"]
    print(f"\n  Read the '{lead}' column of `curvicost audit` first: it is the error type these\n"
          f"  predictions most resemble on this data. Shares are coarse -- read them as\n"
          f"  \"mostly {lead}-like\", not as a measurement.", file=stream)
    if flagged:
        print("\n  + more than 30% of the disagreement is explained by no combination of the five\n"
              "    operators. The commonest cause is over-tracing: only `bridge` adds structure,\n"
              "    so a method that paints more than the reference falls outside the model and the\n"
              "    audit's columns describe it only in part.", file=stream)


def _cmd_perturb(args):
    from .perturb import perturb

    mask, vox = load_mask(args.mask)
    out, info = perturb(mask, args.operator, args.severity, seed=args.seed,
                        prune_px=args.prune_px)
    save_mask(out, args.output, voxel_size=vox)
    if not args.quiet:
        print(f"wrote {args.output}", file=sys.stderr)
        json.dump({k: v for k, v in info.items() if isinstance(v, (int, float, str))},
                  sys.stderr, indent=2, default=str)
        print(file=sys.stderr)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="curvicost",
        description="Score curvilinear segmentations against the functional cost "
                    "of their errors, not just overlap.",
    )
    parser.add_argument("--version", action="version", version=f"curvicost {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    s = subs.add_parser("score", help="score a prediction against a reference")
    s.add_argument("prediction", help="predicted binary mask")
    s.add_argument("--gt", "--reference", required=True, dest="gt",
                   help="reference binary mask")
    s.add_argument("--json", metavar="PATH", help="write the full row as JSON")
    s.add_argument("--csv", metavar="PATH", help="write the full row as one CSV row")
    s.add_argument("--no-erl", action="store_true",
                   help="skip ERL and the DIADEM-like score (roughly halves runtime)")
    s.add_argument("--prune-px", type=int, default=5, metavar="N",
                   help="spur-pruning length in voxels (default: 5)")
    s.add_argument("--voxel-size", metavar="SPEC",
                   help='physical voxel size, e.g. "0.5,0.5,2"; reported, not applied')
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=_cmd_score)

    p = subs.add_parser("perturb", help="generate a graded, typed error")
    p.add_argument("mask", help="input binary mask")
    p.add_argument("--operator", required=True,
                   choices=["break", "bridge", "truncate", "radius", "boundary"])
    p.add_argument("--severity", required=True, type=float,
                   help="fraction of the eligible population (break/bridge/truncate), "
                        "fraction of foreground voxels (boundary), "
                        "or a scale factor (radius, e.g. 0.7)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prune-px", type=int, default=5, metavar="N")
    p.add_argument("-o", "--output", required=True, help="where to write the result")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_perturb)

    pr = subs.add_parser("profile", help="which error types does your method make?")
    pr.add_argument("predictions", nargs="+", help="predicted masks from the method under test")
    pr.add_argument("--gt", "--reference", required=True, dest="gt", help="reference binary mask")
    pr.add_argument("--json", metavar="PATH", help="write the full profile as JSON")
    pr.add_argument("--prune-px", type=int, default=5, metavar="N")
    pr.add_argument("--seed", type=int, default=0, help="first seed for the calibration perturbations")
    pr.add_argument("--seeds", type=int, default=3, metavar="N",
                    help="realisations of each operator to average into the basis (default 3)")
    pr.add_argument("--quiet", action="store_true")
    pr.set_defaults(func=_cmd_profile)

    a = subs.add_parser(
        "audit", help="which error types is your metric blind to?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Perturb each reference mask with every operator across a severity ladder, "
            "score every case, and correlate each metric against each cost per operator.\n"
            "\n"
            "The DEFAULT LADDER IS SHORTENED relative to the accompanying study, because an\n"
            "audit is a diagnostic run on your own machine:\n"
            "  default  break/bridge/truncate at 2, 5, 10, 20, 50 % (5 rungs); radius x0.7,\n"
            "           0.85, 1.15, 1.3 (4 factors); boundary 0.5, 2, 5, 10, 20 % (5 levels);\n"
            "           1 seed; 1,000 bootstrap iterations.\n"
            "  study    break/bridge/truncate at 1, 2, 5, 10, 20, 35, 50 % (7 rungs); radius\n"
            "           x0.5, 0.7, 0.85, 1.15, 1.3; boundary 0.2, 0.5, 1, 2, 5, 10, 20 %;\n"
            "           3 seeds; 2,000 bootstrap iterations.  Reproduce it with --study-ladder.\n"
            "--severities widens break/bridge/truncate ONLY; use --radius-scales and\n"
            "--boundary-fracs for the other two operators.\n"
            "\n"
            "Each (cost, operator, metric) cell gets one of the paper's four labels:\n"
            "  undefined       the cost or the metric is constant across the cell\n"
            "  inconclusive    the operator acted in < 10 units or < 12 cases (rho is shown,\n"
            "                  but is not a validated correlation; never read as blind/anti)\n"
            "  blind           the 95% cluster-bootstrap interval includes zero\n"
            "  anti-correlated sign-aligned rho < 0 and the interval excludes zero\n"
            "Units are reference masks, so a cell can only leave 'inconclusive' with 10+ masks."
        ))
    a.add_argument("masks", nargs="+",
                   help="reference binary masks (3+ for intervals, 10+ for any verdict)")
    a.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    a.add_argument("--csv", metavar="PATH", help="write the findings table as CSV")
    a.add_argument("--cost", action="append", metavar="NAME",
                   choices=["traceable_frac", "conductance_twosided", "conductance_frac",
                            "traceable_single_frac"],
                   help="cost to correlate against; repeatable "
                        "(default: traceable_frac and conductance_twosided)")
    a.add_argument("--min-units", type=int, default=None, metavar="N",
                   help="a cell in which the operator acted in fewer than N reference masks "
                        "is inconclusive (default 10, the paper's floor)")
    a.add_argument("--min-cases", type=int, default=None, metavar="N",
                   help="a cell with fewer than N acting cases is inconclusive "
                        "(default 12, the paper's minimum group size)")
    a.add_argument("--severities", metavar="LIST",
                   help="comma-separated severities for break/bridge/truncate ONLY "
                        "(default 0.02,0.05,0.1,0.2,0.5; study 0.01,0.02,0.05,0.1,0.2,0.35,0.5)")
    a.add_argument("--radius-scales", metavar="LIST",
                   help="comma-separated scale factors for radius "
                        "(default 0.7,0.85,1.15,1.3; study 0.5,0.7,0.85,1.15,1.3)")
    a.add_argument("--boundary-fracs", metavar="LIST",
                   help="comma-separated fractions of foreground voxels for boundary "
                        "(default 0.005,0.02,0.05,0.1,0.2; study 0.002,0.005,0.01,0.02,0.05,0.1,0.2)")
    a.add_argument("--seeds", type=int, default=None, metavar="N",
                   help="random seeds per operator/severity (default 1; study 3)")
    a.add_argument("--n-boot", "--bootstrap", dest="n_boot", type=int, default=None, metavar="N",
                   help="cluster-bootstrap iterations (default 1000; study 2000)")
    a.add_argument("--study-ladder", action="store_true",
                   help="use the study's full ladder, 3 seeds and 2,000 bootstrap iterations "
                        "(explicit --severities/--radius-scales/--boundary-fracs/--seeds/"
                        "--n-boot still override)")
    a.add_argument("--no-erl", action="store_true")
    a.add_argument("--prune-px", type=int, default=5, metavar="N")
    a.add_argument("--include-cases", action="store_true",
                   help="also write every scored case into the JSON")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(func=_cmd_audit)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"curvicost: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
