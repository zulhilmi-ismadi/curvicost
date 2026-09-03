"""Score a curvilinear segmentation against both overlap metrics and functional cost.

The point of this package is the second column. Dice tells you how many voxels
you got; it does not tell you what the errors cost the biology. `score()`
returns both, from one pass, so the two can be compared on the same case.

Two invariants are load-bearing and must not be "optimised" away:

1. **The reference and the prediction go through the same path.** Both are
   skeletonised and re-graphed from their own mask. Nothing is inherited from
   the reference graph. If cost were read off the reference graph, thinning
   every vessel by 30% would cost nothing -- which is false, since Poiseuille
   conductance goes as r^4.

2. **Sinks are fixed to the reference's terminal locations.** Without this,
   conductance is summed over whatever leaves the *predicted* graph happens to
   have, which rewards fragmentation: shattering a tree manufactures new
   low-resistance leaves near the source and the score goes UP.
"""
from __future__ import annotations

from skimage.morphology import skeletonize

from ._core.pipeline import analyse
from ._core import metrics as _metrics
from ._core.erl import erl_from_masks, diadem_like_score

__all__ = ["score", "METRIC_COLUMNS", "COST_COLUMNS"]

METRIC_COLUMNS = (
    "dice", "iou", "cldice", "betti0_gt", "betti0_pred", "betti0_error",
    "erl", "erl_frac", "merged_frac", "diadem_like",
)
COST_COLUMNS = (
    "traceable_frac", "conductance_frac", "perfused_of_self",
    "traceable_length", "conductance", "traceable_length_ref", "conductance_ref",
)


def score(gt, pred, *, with_erl=True, prune_px=5, reference_skeleton=None):
    """Score `pred` against `gt`. Returns a flat dict of metrics and costs.

    Parameters
    ----------
    gt, pred : ndarray of bool
        2D or 3D binary masks of the same shape. Anything non-zero is
        foreground.
    with_erl : bool
        Compute expected run length and the DIADEM-like score. ERL labels
        components twice and roughly doubles per-case cost, so it can be
        switched off for large sweeps. Default on: ERL is the metric this
        work recommends, and leaving it out of the default would be perverse.
    prune_px : int
        Spur-pruning length, in voxels, applied to both skeleton graphs.
    reference_skeleton : ndarray, optional
        Precomputed `skeletonize(gt)`. Pass it when scoring many predictions
        against one reference; it is constant per reference.

    Notes
    -----
    Cost *fractions* are relative to the reference's own cost, so 1.0 means
    "this prediction preserves the reference's function" and 0.0 means it
    preserves none of it. Values above 1.0 are possible and meaningful: a
    prediction more liberal than the reference can be more traceable than it
    (STARE's two annotators differ by traceable_frac 1.95).
    """
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    if gt.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got {gt.ndim}D")

    gt = gt.astype(bool)
    pred = pred.astype(bool)
    skel = reference_skeleton if reference_skeleton is not None else skeletonize(gt)

    # Reference first: its terminal positions become the fixed perfusion sinks.
    g_ref, _, c_ref = analyse(gt, prune_px=prune_px)
    g_pred, _, c_pred = analyse(pred, prune_px=prune_px,
                                sink_positions=c_ref["sink_positions"])

    out = _metrics.all_metrics(gt, pred, gt_skel=skel, with_erl=with_erl)
    if with_erl:
        # g_pred is reused rather than rebuilt: rebuilding the graph here is
        # invisible in 2D at 0.2 s and dominant on 256^3 volumes.
        out["diadem_like"] = diadem_like_score(g_ref, g_pred) if g_pred is not None else 1.0

    t_ref = c_ref["traceable_length"]
    k_ref = c_ref["conductance"]
    out.update(
        traceable_length=c_pred["traceable_length"],
        conductance=c_pred["conductance"],
        traceable_length_ref=t_ref,
        conductance_ref=k_ref,
        traceable_frac=c_pred["traceable_length"] / t_ref if t_ref > 0 else float("nan"),
        conductance_frac=c_pred["conductance"] / k_ref if k_ref > 0 else float("nan"),
        perfused_of_self=c_pred["perfused_of_self"],
    )
    return out
