"""`curvicost audit` — which error types is your metric blind to, on YOUR data?

The paper this package accompanies makes one claim a user cannot act on from
a published table: whether a metric predicts functional cost depends on the
error type, the structure class AND the cost, so the answer for retina does
not transfer to neurites. `audit` therefore recomputes the whole map on the
user's own reference masks: perturb with each operator across a severity
ladder, score every case, and correlate each metric against each cost,
separately per operator.

What comes back is a blindness report. A metric with a correlation whose CI
includes zero for some operator cannot see that error type on this data --
report it and you are reporting nothing about that failure mode.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

from .perturb import perturb, OPERATORS
from .score import score, prepare_reference

__all__ = ["audit", "sweep", "scale_report", "DEFAULT_SEVERITIES", "COSTS"]

#: Kept short by default: an audit is a diagnostic the user runs on their own
#: machine, not the paper's 9,612-case sweep. Widen with --severities.
DEFAULT_SEVERITIES = {
    "break":    (0.02, 0.05, 0.10, 0.20, 0.50),
    "bridge":   (0.02, 0.05, 0.10, 0.20, 0.50),
    "truncate": (0.02, 0.05, 0.10, 0.20, 0.50),
    "radius":   (0.70, 0.85, 1.15, 1.30),
    "boundary": (0.005, 0.02, 0.05, 0.10, 0.20),
}
COSTS = ("traceable_frac", "conductance_twosided")

#: Below this many clustering units a percentile interval over resampled units is
#: not informative -- with four masks a bootstrap has at most 35 distinct resamples,
#: so "the interval includes zero" says more about the sample than about the metric.
#: Findings below the floor are reported as `inconclusive` rather than `blind`.
MIN_UNITS_FOR_BLINDNESS = 10

#: False where a higher value means a WORSE segmentation, so its correlation
#: must be negated before being read as "tracks the cost".
HIGHER_IS_BETTER = {
    "dice": True, "iou": True, "cldice": True, "betti0_error": False,
    "erl_frac": True, "diadem_like": True,
}
METRIC_ORDER = ("dice", "iou", "cldice", "betti0_error", "erl_frac", "diadem_like")


def scale_report(mask, prune_px=5):
    """How big is `prune_px` in units of this mask's own vessel width?

    `prune_px` is an ABSOLUTE pixel length, so the same value does different
    jobs on differently-sampled data. On STARE (median skeleton radius 2.1 px)
    the default 5 prunes spurs shorter than ~2.4 vessel radii; on FIVES (radius
    6.3 px) the same 5 prunes only ~0.8 radii, keeping far more small branches.
    Correlations are not comparable across datasets unless this ratio is.

    Returns {"median_radius", "prune_px", "prune_in_radii"}.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    mask = np.asarray(mask).astype(bool)
    skel = skeletonize(mask)
    dist = ndimage.distance_transform_edt(mask)
    radii = dist[skel]
    median = float(np.median(radii)) if radii.size else float("nan")
    return dict(median_radius=median, prune_px=prune_px,
                prune_in_radii=(prune_px / median) if median > 0 else float("nan"))


def sweep(mask, *, severities=None, seeds=(0,), prune_px=5, with_erl=True,
          unit="mask", on_case=None):
    """Perturb `mask` with every operator and score each case.

    Yields one dict per case. The `radius` operator is scored against its own
    scale-1.0 reconstruction rather than the original mask: rasterising a
    thickness change is itself lossy, and charging that loss to the operator
    would make a purely geometric error look like a topological one.
    """
    severities = severities or DEFAULT_SEVERITIES
    mask = np.asarray(mask).astype(bool)
    radius_ref = radius_ctx = None
    ctx = prepare_reference(mask, prune_px=prune_px, with_erl=with_erl)
    rows = []

    for operator in OPERATORS:
        for sev in severities.get(operator, ()):
            for seed in (seeds if operator != "radius" else (seeds[0],)):
                try:
                    out, info = perturb(mask, operator, sev, seed=seed,
                                        prune_px=prune_px)
                except ValueError:
                    continue
                if operator == "radius":
                    if radius_ref is None:
                        radius_ref, _ = perturb(mask, "radius", 1.0, prune_px=prune_px)
                        radius_ctx = prepare_reference(radius_ref, prune_px=prune_px,
                                                       with_erl=with_erl)
                    reference, ref_ctx = radius_ref, radius_ctx
                else:
                    reference, ref_ctx = mask, ctx
                if np.array_equal(out, reference):
                    continue          # operator declined; not a data point
                row = score(reference, out, with_erl=with_erl, prune_px=prune_px,
                            reference=ref_ctx)
                # What the operator actually did, so the audit can tell a real
                # severity ladder from one that saturated on a small population.
                realised = info.get("n_broken", info.get("n_bridges",
                           info.get("n_truncated", info.get("voxels_moved", None))))
                row.update(unit=unit, operator=operator, severity=float(sev),
                           seed=int(seed), realised=realised,
                           population=info.get("population", None))
                rows.append(row)
                if on_case is not None:
                    on_case(row)
    return rows


def _bootstrap(x, y, units, n_boot, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    units = np.asarray(units)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, units = x[ok], y[ok], units[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    obs = spearmanr(x, y).statistic
    uniq = list(dict.fromkeys(units.tolist()))
    if len(uniq) < 3:
        return obs, np.nan, np.nan       # too few units for an honest CI
    groups = [np.where(units == u)[0] for u in uniq]
    rng = np.random.default_rng(seed)
    draws = []
    # A resample can legitimately draw the same unit throughout and leave one
    # array constant. That draw is simply undefined and is dropped; it is not
    # a problem worth warning the user about on every audit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            r = spearmanr(x[idx], y[idx]).statistic
            if np.isfinite(r):
                draws.append(r)
    if not draws:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def audit(rows, *, costs=COSTS, n_boot=1000, seed=0, attempted=None,
          min_units=MIN_UNITS_FOR_BLINDNESS):
    """Correlate every metric against every cost, per operator.

    `attempted` names the operators that were actually swept. It cannot be
    inferred from `rows`, because an operator that declined on every severity
    and one that was never requested both leave no trace there -- and the
    first is worth reporting while the second is not.

    Returns {"n_cases", "n_units", "has_ci", "findings": [...]}. Each finding
    carries the SIGN-ALIGNED rho, so "more positive" always means "this metric
    tracks this cost", whichever direction the raw metric runs.

    `min_units` is the floor below which a wide interval is reported as
    `inconclusive` instead of `blind`: absence of evidence is not evidence of
    blindness, and on a handful of masks every interval is wide.
    """
    attempted = tuple(OPERATORS if attempted is None else attempted)
    notes = {}                       # operator -> list of reasons; joined on return

    def note(operator, why):
        if why not in notes.setdefault(operator, []):
            notes[operator].append(why)

    for operator in attempted:
        # Explain every operator that cannot produce a number. A blank column
        # with no reason is indistinguishable from a bug, and one of these
        # reasons IS effectively a bug in the caller's setup.
        n = sum(1 for r in rows if r["operator"] == operator)
        if n == 0:
            note(operator, "produced no cases — the operator declined on every "
                           "severity. `bridge` needs two skeleton tips within "
                           "max_gap (12 px, an ABSOLUTE length) but far apart in "
                           "the graph; on thick or heavily pruned structures no "
                           "such pair exists.")
        elif n < 3:
            note(operator, f"only {n} case(s) — needs 3+ to correlate. Pass more "
                           "values to --severities or raise --seeds.")
    joined = lambda: {op: " ".join(why) for op, why in notes.items()}
    if not rows:
        return dict(n_cases=0, n_units=0, has_ci=False, findings=[], notes=joined())
    units = sorted({r["unit"] for r in rows})
    metrics = [m for m in METRIC_ORDER if m in rows[0]]
    findings = []
    for cost in costs:
        for operator in OPERATORS:
            sub = [r for r in rows if r["operator"] == operator]
            if len(sub) < 3:
                continue
            cost_vals = np.asarray([s[cost] for s in sub], float)
            cost_vals = cost_vals[np.isfinite(cost_vals)]
            # A cost the operator does not move (reachable length under `bridge`:
            # every reference voxel stays reachable, the fraction sits at 1.0) makes
            # every correlation undefined. That is a statement about the COST, not
            # about any metric, so it is reported as its own category rather than
            # as blindness.
            cost_constant = len(np.unique(cost_vals)) < 2
            # Severity is a fraction of the operator's eligible population. On a
            # small population the ladder saturates -- every severity rounds to
            # the same count -- and the "cases" are one mask scored repeatedly.
            realised = [(r["unit"], r.get("realised")) for r in sub if r.get("realised") is not None]
            distinct = len(set(realised))
            # Flag only real saturation (half or more of the cases are repeats), not
            # the odd coincidence of two boundary draws moving the same voxel count.
            if realised and distinct <= len(sub) // 2:
                units_with = sorted({u for u, _ in realised})
                note(operator, f"{len(sub)} cases but only {distinct} distinct realisation(s): "
                               f"the severity ladder saturated on a small eligible population "
                               f"(units with cases: {', '.join(str(u).rsplit('/', 1)[-1] for u in units_with)}). "
                               "Correlations over repeated copies of one mask are not evidence.")
            if cost_constant:
                note(operator, f"{cost} did not move under this operator (constant at "
                               f"{cost_vals[0]:.3g}), so every correlation against it is "
                               "undefined. This is a property of the cost, not of any "
                               "metric: a cost that an error type cannot change is the "
                               "wrong cost for that error type." if len(cost_vals) else
                               f"{cost} is undefined on every case for this operator.")
            for metric in metrics:
                r, lo, hi = _bootstrap([s[metric] for s in sub],
                                       [s[cost] for s in sub],
                                       [s["unit"] for s in sub], n_boot, seed)
                if not HIGHER_IS_BETTER.get(metric, True) and np.isfinite(r):
                    r, lo, hi = -r, (-hi if np.isfinite(hi) else np.nan), \
                                (-lo if np.isfinite(lo) else np.nan)
                undefined = not np.isfinite(r)
                if undefined and not cost_constant:
                    note(operator, f"{metric} was constant under this operator, so its "
                                   "correlation is undefined — the perturbation did not "
                                   "move that metric at all.")
                spans_zero = (not undefined) and np.isfinite(lo) and lo < 0 < hi
                # Too few units to tell "no relation" from "not enough data".
                inconclusive = spans_zero and len(units) < min_units
                blind = spans_zero and not inconclusive
                findings.append(dict(
                    cost=cost, operator=operator, metric=metric, n=len(sub),
                    aligned_rho=r, ci_lo=lo, ci_hi=hi, blind=bool(blind),
                    inconclusive=bool(inconclusive),
                    anti=bool(np.isfinite(r) and r < -0.2 and not spans_zero),
                    undefined=bool(undefined), cost_constant=bool(cost_constant)))
    if 0 < len(units) < min_units:
        note("__sample__", f"{len(units)} clustering unit(s): below {min_units}, a percentile interval over "
                           "resampled units cannot separate 'no relation' from 'not enough data', so cells whose "
                           "interval spans zero are reported as inconclusive rather than blind.")
    return dict(n_cases=len(rows), n_units=len(units), has_ci=len(units) >= 3,
                min_units=min_units, findings=findings, notes=joined())
