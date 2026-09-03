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
_COST_ORDER = ("traceable_frac", "conductance_frac", "perfused_of_self")


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _report(row, stream):
    """Human-readable output: metrics beside costs, which is the whole point."""
    print("metric                 value   |  cost                    value", file=stream)
    print("-" * 66, file=stream)
    left = [(k, row[k]) for k in _METRIC_ORDER if k in row]
    right = [(k, row[k]) for k in _COST_ORDER if k in row]
    for i in range(max(len(left), len(right))):
        lk, lv = left[i] if i < len(left) else ("", "")
        rk, rv = right[i] if i < len(right) else ("", "")
        lcell = f"{lk:<18} {_fmt(lv):>8}" if lk else " " * 27
        rcell = f"{rk:<20} {_fmt(rv):>8}" if rk else ""
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


_COST_LABEL = {"traceable_frac": "traceable length", "conductance_frac": "conductance"}


def _audit_report(result, stream):
    """The blindness report: metrics down, operators across, one block per cost."""
    from .audit import OPERATORS, METRIC_ORDER

    metrics = [m for m in METRIC_ORDER
               if any(f["metric"] == m for f in result["findings"])]
    print(f"\n{result['n_cases']} cases from {result['n_units']} reference mask(s)",
          file=stream)
    if not result["has_ci"]:
        print("⚠  fewer than 3 reference masks: correlations are reported WITHOUT "
              "confidence\n   intervals, and blindness cannot be judged. Treat these as "
              "indicative only.", file=stream)

    for cost in dict.fromkeys(f["cost"] for f in result["findings"]):
        print(f"\n  vs {_COST_LABEL.get(cost, cost)}", file=stream)
        print("  " + "metric".ljust(14) + "".join(o.rjust(11) for o in OPERATORS),
              file=stream)
        for m in metrics:
            cells = ""
            for op in OPERATORS:
                f = next((f for f in result["findings"]
                          if f["cost"] == cost and f["operator"] == op
                          and f["metric"] == m), None)
                if f is None or f["aligned_rho"] != f["aligned_rho"]:
                    cells += "         --"
                else:
                    mark = "*" if f["blind"] else ("!" if f["anti"] else " ")
                    cells += f"{f['aligned_rho']:>+10.2f}{mark}"
            print("  " + m.ljust(14) + cells, file=stream)

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
    if not result["findings"]:
        print("\nNo correlations could be computed. See above.", file=stream)
        return
    blind = [f for f in result["findings"] if f["blind"]]
    anti = [f for f in result["findings"] if f["anti"]]
    print(f"\n  * CI includes zero — blind to that error type   ({len(blind)} of "
          f"{len(result['findings'])} combinations)", file=stream)
    print(f"  ! anti-correlates — the metric moves the WRONG WAY   ({len(anti)})",
          file=stream)
    if anti:
        worst = min(anti, key=lambda f: f["aligned_rho"])
        print(f"    worst: {worst['metric']} vs {_COST_LABEL.get(worst['cost'])} on "
              f"{worst['operator']} (ρ {worst['aligned_rho']:+.2f}). A cost that a "
              f"perturbation\n    can IMPROVE is the wrong cost for that error type — "
              "no metric choice fixes it.", file=stream)


def _cmd_audit(args):
    import json as _json

    import numpy as np
    from .audit import sweep, audit, DEFAULT_SEVERITIES

    severities = dict(DEFAULT_SEVERITIES)
    if args.severities:
        vals = tuple(float(v) for v in args.severities.split(","))
        for op in ("break", "bridge", "truncate"):
            severities[op] = vals

    from .audit import scale_report

    rows, seeds, scales = [], tuple(range(args.seeds)), []
    for path in args.masks:
        mask, _ = load_mask(path)
        if not args.quiet:
            print(f"sweeping {path} ...", file=sys.stderr, flush=True)
        scales.append(scale_report(mask, args.prune_px))
        rows += sweep(mask, severities=severities, seeds=seeds,
                      prune_px=args.prune_px, with_erl=not args.no_erl,
                      unit=str(path))
    attempted = [op for op in severities if severities.get(op)]
    result = audit(rows, n_boot=args.n_boot, attempted=attempted)
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
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(result["findings"][0]))
            w.writeheader(); w.writerows(result["findings"])
    if not args.quiet:
        _audit_report(result, sys.stdout)
    return 0


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

    a = subs.add_parser("audit", help="which error types is your metric blind to?")
    a.add_argument("masks", nargs="+", help="reference binary masks (3+ for CIs)")
    a.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    a.add_argument("--csv", metavar="PATH", help="write the findings table as CSV")
    a.add_argument("--severities", metavar="LIST",
                   help="comma-separated severities for break/bridge/truncate "
                        "(default 0.02,0.05,0.1,0.2,0.5)")
    a.add_argument("--seeds", type=int, default=1, metavar="N",
                   help="random seeds per operator/severity (default 1)")
    a.add_argument("--n-boot", type=int, default=1000, metavar="N",
                   help="cluster-bootstrap iterations (default 1000)")
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
