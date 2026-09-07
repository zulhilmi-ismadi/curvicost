"""Score a curvilinear segmentation against both overlap metrics and functional cost.

The point of this package is the second column. Dice tells you how many voxels
you got; it does not tell you what the errors cost the biology. `score()`
returns both, from one pass, so the two can be compared on the same case.

The cost definitions are the study's (its "v6" data of record, 2026-09-08),
composed here exactly as the study's re-score script composes them:

* **Reachable length** (`traceable_frac`). The reference is skeletonised once.
  Each connected component of the reference is traced from its own root (the
  skeleton voxel of maximum distance-transform value), and the cost counts the
  reference-skeleton voxels that lie in the perturbed mask's connected
  component containing that root. It is bounded by 1: a prediction cannot be
  "more reachable" than the reference, and a spurious bridge between two
  reference fragments adds nothing (their voxels were already counted). The
  single-source variant, traced from one pinned source, is `traceable_single_frac`.
* **Conductance** (`conductance_frac`). A Kirchhoff (resistor-network) solve on
  the prediction's own skeleton graph: unit pressure at the source, zero at every
  surviving reference sink, resistance integrated pixel by pixel from the
  distance transform as 1/r^4. Adding an edge can never lower it (Rayleigh
  monotonicity), so a bridge is not charged as a flow loss.

Four invariants are load-bearing and must not be "optimised" away:

1. **The reference and the prediction go through the same path.** Both are
   skeletonised and re-graphed from their own mask. If cost were read off the
   reference graph, thinning every vessel by 30% would cost nothing -- which is
   false, since Poiseuille conductance goes as r^4.
2. **Sinks are fixed to the reference's terminal locations** and survive only
   through mask connectivity to the source. Without this, conductance is summed
   over whatever leaves the *predicted* graph happens to have, which rewards
   fragmentation.
3. **The source is pinned to the reference's source location.** Re-picking the
   maximum-radius node on every prediction let a break near the source move it
   to another trunk and *raise* conductance.
4. **Reachable length is counted on the reference skeleton**, through the
   prediction's connectivity, never on a re-skeletonised prediction: cut faces
   re-skeletonise into extra length and inflated the old value above 1.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from ._core.pipeline import analyse
from ._core import metrics as _metrics
from ._core.erl import erl_from_masks, diadem_like_score
from ._core.skelgraph import skeleton_to_graph, prune_spurs
from ._core.reach import reach_count, reach_count_multi, ref_roots_for

__all__ = ["score", "prepare_reference", "METRIC_COLUMNS", "COST_COLUMNS"]

METRIC_COLUMNS = (
    "dice", "iou", "cldice", "betti0_gt", "betti0_pred", "betti0_error",
    "erl", "erl_frac", "merged_frac", "diadem_like",
)
COST_COLUMNS = (
    "traceable_frac", "traceable_single_frac", "conductance_frac", "perfused_of_self",
    "traceable_length", "conductance", "traceable_length_ref", "conductance_ref",
)


def prepare_reference(gt, *, prune_px=5, with_erl=True, reference_skeleton=None):
    """Everything `score()` needs from the reference alone, computed once.

    Scoring many predictions against one reference repeats this work for no
    reason; pass the result as `score(..., reference=ctx)`. The context is tied
    to `gt` and `prune_px`; `score()` refuses a context built for another shape.
    """
    gt = np.asarray(gt).astype(bool)
    if gt.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got {gt.ndim}D")
    skel = reference_skeleton if reference_skeleton is not None else skeletonize(gt)
    skel = np.asarray(skel).astype(bool)

    # Graph used by the DIADEM-like score (the study builds it without a radius map).
    g_metric, _ = skeleton_to_graph(gt, skel=skel)
    g_metric, _ = prune_spurs(g_metric, prune_px)

    # The reference's own terminals become the fixed sinks; the reference is then
    # analysed against them so its conductance is measured under the same sink
    # rule as every prediction.
    _, _, c00 = analyse(gt, prune_px=prune_px)
    sinks = c00.get("sink_positions", np.zeros((0, gt.ndim)))
    g_ref, src, c0 = analyse(gt, prune_px=prune_px, sink_positions=sinks)
    if g_ref is None or src is None:
        src_pos = None
    else:
        src_pos = np.asarray(g_ref.nodes[src]["pos"], dtype=float)

    edt = ndimage.distance_transform_edt(gt)
    ref_lab, roots = ref_roots_for(gt, skel, edt)
    base_multi = reach_count_multi(gt, skel, ref_lab, roots)
    base_single = reach_count(gt, skel, src_pos) if src_pos is not None else 0

    erl_ref = None
    if with_erl:
        erl_ref, _ = erl_from_masks(gt, gt, gt_skel=skel)

    return dict(
        shape=gt.shape, prune_px=prune_px, skel=skel, g_metric=g_metric,
        sinks=sinks, src_pos=src_pos,
        source_positions_all=c0.get("source_positions_all"),
        conductance_ref=float(c0.get("conductance_k", 0.0)),
        ref_lab=ref_lab, roots=roots,
        base_multi=int(base_multi), base_single=int(base_single),
        erl_ref=erl_ref,
    )


def score(gt, pred, *, with_erl=True, prune_px=5, reference_skeleton=None,
          reference=None):
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
        Precomputed `skeletonize(gt)`. Ignored when `reference` is given.
    reference : dict, optional
        Output of `prepare_reference(gt, ...)`. Pass it when scoring many
        predictions against one reference.

    Notes
    -----
    Cost *fractions* are relative to the reference's own cost, so 1.0 means
    "this prediction preserves the reference's function" and 0.0 means it
    preserves none of it. `traceable_frac` cannot exceed 1.0 (it counts
    reference voxels). `conductance_frac` can: a prediction thicker than the
    reference conducts more.
    """
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    if gt.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got {gt.ndim}D")

    if reference is None:
        reference = prepare_reference(gt, prune_px=prune_px, with_erl=with_erl,
                                      reference_skeleton=reference_skeleton)
    elif tuple(reference["shape"]) != gt.shape or reference["prune_px"] != prune_px:
        raise ValueError("`reference` was prepared for a different mask or prune_px")
    ctx = reference
    skel = ctx["skel"]

    out = _metrics.all_metrics(gt, pred, gt_skel=skel, with_erl=with_erl,
                               erl_ref=ctx["erl_ref"])

    # Prediction: pinned source, fixed sinks, the reference's component roots.
    g_pred, _, c = analyse(pred, prune_px=prune_px, sink_positions=ctx["sinks"],
                           source_pos=ctx["src_pos"],
                           source_positions_all=ctx["source_positions_all"])
    if with_erl:
        # g_pred is reused rather than rebuilt: rebuilding the graph here is
        # invisible in 2D at 0.2 s and dominant on 256^3 volumes.
        out["diadem_like"] = diadem_like_score(ctx["g_metric"], g_pred) if g_pred is not None else 1.0

    reach_multi = reach_count_multi(pred, skel, ctx["ref_lab"], ctx["roots"])
    reach_single = (reach_count(pred, skel, ctx["src_pos"])
                    if ctx["src_pos"] is not None else 0)
    cond = float(c.get("conductance_k", 0.0))
    k_ref = ctx["conductance_ref"]
    out.update(
        traceable_length=int(reach_multi),
        conductance=cond,
        traceable_length_ref=ctx["base_multi"],
        conductance_ref=k_ref,
        traceable_frac=reach_multi / ctx["base_multi"] if ctx["base_multi"] > 0 else float("nan"),
        traceable_single_frac=reach_single / ctx["base_single"] if ctx["base_single"] > 0 else float("nan"),
        conductance_frac=cond / k_ref if k_ref > 0 else float("nan"),
        perfused_of_self=float(c.get("perfused_of_self", 0.0)),
    )
    return out
