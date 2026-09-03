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
