"""`curvicost profile` — which of the five error types does YOUR method actually make?

The audit tells you which metric tracks which cost for each error type. Acting on it
needs one more thing the audit cannot supply: knowing which error types your own
method produces. Without that, "stratify by error type" is advice a user cannot
follow, because a real prediction is a mixture and arrives unlabelled.

The method here is deliberately modest. Each (reference, prediction) pair is reduced
to a signature of seven interpretable quantities, all of which the package already
computes. The same signature is then measured for each of the five operators applied
to *the user's own reference*, at the severity whose overall disagreement matches the
prediction's, so the comparison is calibrated to this data rather than to ours. The
observed signature is expressed as a non-negative combination of the five operator
signatures, and the weights are reported as shares.

What that is: a coarse, per-dataset answer to "which columns of the audit apply to
me". What it is not: a segmentation of the error into causes. Shares are reported to
the nearest per cent and should be read as "mostly break-like", not as a measurement.
`residual` says how much of the signature the five operators failed to explain; a
large residual means the prediction does something the operator set does not model,
which is itself worth knowing, and the most common cause is over-tracing, since only
`bridge` adds structure.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import nnls
from skimage.morphology import skeletonize

from ._core import metrics as _metrics
from ._core.skelgraph import skeleton_to_graph, prune_spurs
from ._core.reach import reach_count_multi, ref_roots_for
from .perturb import perturb, OPERATORS

__all__ = ["profile", "signature", "SIGNATURE_FIELDS"]

#: The signature. Each entry is a quantity the operators move differently:
#: removals and additions separate loss from gain; the component change separates
#: break (raises it) from bridge (lowers it); merged fraction fires only on merges;
#: reachable-length loss weights a removal by how much arbor it orphans; the radius
#: ratio catches uniform thickening or thinning, which moves no topology; and the
#: terminal ratio catches shortening, which removes free ends without splitting.
SIGNATURE_FIELDS = ("removed", "added", "component_change", "merged",
                    "reach_loss", "radius_ratio", "terminal_loss")


def _graph_stats(mask, prune_px):
    skel = skeletonize(mask)
    if not skel.any():
        return 0, 1.0
    g, _ = skeleton_to_graph(mask, skel=skel)
    g, _ = prune_spurs(g, prune_px)
    terminals = sum(1 for n in g.nodes if g.degree(n) == 1)
    dist = ndimage.distance_transform_edt(mask)
    radii = dist[skel]
    return terminals, (float(np.median(radii)) if radii.size else 1.0)


def signature(gt, pred, *, prune_px=5, _ref=None):
    """Seven interpretable quantities describing how `pred` differs from `gt`."""
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    n_gt = max(int(gt.sum()), 1)
    if _ref is None:
        skel = skeletonize(gt)
        ref_lab, roots = ref_roots_for(gt, skel, ndimage.distance_transform_edt(gt))
        base = max(reach_count_multi(gt, skel, ref_lab, roots), 1)
        b0 = _metrics.betti0(gt)
        term, rad = _graph_stats(gt, prune_px)
        _ref = dict(skel=skel, ref_lab=ref_lab, roots=roots, base=base, b0=b0, term=term, rad=rad)
    term_p, rad_p = _graph_stats(pred, prune_px)
    reach = reach_count_multi(pred, _ref["skel"], _ref["ref_lab"], _ref["roots"]) / _ref["base"]
    return np.array([
        float((gt & ~pred).sum()) / n_gt,                                   # removed
        float((pred & ~gt).sum()) / n_gt,                                   # added
        (_metrics.betti0(pred) - _ref["b0"]) / max(_ref["b0"], 1),          # component_change
        float(_metrics.all_metrics(gt, pred, gt_skel=_ref["skel"], with_erl=True)["merged_frac"]),
        1.0 - reach,                                                        # reach_loss
        rad_p / max(_ref["rad"], 1e-9) - 1.0,                               # radius_ratio
        1.0 - term_p / max(_ref["term"], 1),                                # terminal_loss
    ], dtype=float), _ref


#: Scales that put the seven quantities on comparable footing. They are fixed rather
#: than fitted, so a profile does not change when the calibration set does.
_SCALE = np.array([0.20, 0.20, 1.00, 0.10, 0.20, 0.20, 0.30])

CALIBRATION = {
    "break":    (0.02, 0.05, 0.10, 0.20, 0.35, 0.50),
    "bridge":   (0.02, 0.05, 0.10, 0.20, 0.50),
    "truncate": (0.02, 0.05, 0.10, 0.20, 0.35, 0.50),
    "radius":   (0.70, 0.85, 1.15, 1.30),
    "boundary": (0.005, 0.02, 0.05, 0.10, 0.20),
}


def profile(gt, pred, *, prune_px=5, seed=0, seeds=3, calibration=None):
    """Express the disagreement between `gt` and `pred` as shares of the five operators.

    Returns {"shares", "residual", "nearest", "observed", "basis", "severities"}.
    Shares are non-negative and sum to at most 1; `residual` is the remainder, the
    part of the signature the operator set cannot account for.

    `seeds` sets how many realisations of each operator the basis averages over,
    starting at `seed`. One realisation is not the operator: which edges `break`
    happens to cut moves the reachable-length loss by a factor of five at fixed
    severity, and a basis built from a single draw makes the fit chase that draw
    rather than the operator. Three is enough to stabilise it and keeps a profile
    to well under a minute on a fundus image.
    """
    calibration = calibration or CALIBRATION
    obs, ref = signature(gt, pred, prune_px=prune_px)
    magnitude = obs[0] + obs[1]                       # total disagreement, removals plus additions

    basis, used_sev, names = [], {}, []
    for op in OPERATORS:
        best, best_gap, best_sev = None, np.inf, None
        for sev in calibration.get(op, ()):
            sigs = []
            for k in range(max(int(seeds), 1)):
                try:
                    out, _ = perturb(gt, op, sev, seed=seed + k, prune_px=prune_px)
                except ValueError:
                    continue
                if np.array_equal(out, gt):
                    continue
                sigs.append(signature(gt, out, prune_px=prune_px, _ref=ref)[0])
            if not sigs:
                continue
            sig = np.mean(sigs, axis=0)               # the operator, not one draw of it
            gap = abs((sig[0] + sig[1]) - magnitude)  # match the operator to this pair's severity
            if gap < best_gap:
                best, best_gap, best_sev = sig, gap, sev
        if best is not None:
            basis.append(best); names.append(op); used_sev[op] = best_sev
    if not basis:
        return dict(shares={}, residual=1.0, nearest=None, observed=dict(zip(SIGNATURE_FIELDS, obs)),
                    basis={}, severities={})

    B = (np.vstack(basis) / _SCALE).T                  # (features, operators)
    y = obs / _SCALE
    w, _ = nnls(B, y)
    total = float(w.sum())
    shares = {n: float(v / total) for n, v in zip(names, w)} if total > 0 else {n: 0.0 for n in names}
    resid = float(np.linalg.norm(y - B @ w) / max(np.linalg.norm(y), 1e-9))
    nearest = max(shares, key=shares.get) if shares else None
    return dict(shares=shares, residual=min(resid, 1.0), nearest=nearest,
                observed=dict(zip(SIGNATURE_FIELDS, obs.tolist())),
                basis={n: dict(zip(SIGNATURE_FIELDS, b.tolist())) for n, b in zip(names, basis)},
                severities=used_sev)
