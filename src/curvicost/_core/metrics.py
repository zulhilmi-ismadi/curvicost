"""Segmentation metrics: one overlap, one topology-aware, one pure topology."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

_FULL = lambda ndim: np.ones((3,) * ndim)


def dice(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def iou(gt, pred):
    union = np.logical_or(gt, pred).sum()
    return 1.0 if union == 0 else float(np.logical_and(gt, pred).sum() / union)


def betti0(mask):
    """Number of connected components (26-connectivity in 3D, 8 in 2D)."""
    _, n = ndimage.label(mask, structure=_FULL(mask.ndim))
    return int(n)


def betti0_error(gt, pred):
    return abs(betti0(pred) - betti0(gt))


def cldice(gt, pred):
    """Shit et al. 2021. Skeleton of one against the volume of the other.

    Uses a hard skeleton; the soft-skeleton variant is skeletoniser-sensitive
    (lepinay-2026), which is a supplement, not the prototype's problem.
    """
    s_pred, s_gt = skeletonize(pred), skeletonize(gt)
    t_prec = _frac(s_pred, gt)
    t_sens = _frac(s_gt, pred)
    denom = t_prec + t_sens
    return 0.0 if denom == 0 else float(2 * t_prec * t_sens / denom)


def _frac(skel, volume):
    n = skel.sum()
    return 1.0 if n == 0 else float(np.logical_and(skel, volume).sum() / n)


def all_metrics(gt, pred, gt_skel=None, with_erl=False, erl_ref=None):
    """Metric suite. ERL is opt-in: it labels components twice and roughly
    doubles per-case cost, so sweeps enable it explicitly.

    `erl_ref` is the intact-case ERL for this image. It is constant per image,
    so callers pass it in rather than have every case recompute it.
    """
    out = {
        "dice": dice(gt, pred),
        "iou": iou(gt, pred),
        "cldice": cldice(gt, pred),
        "betti0_gt": betti0(gt),
        "betti0_pred": betti0(pred),
        "betti0_error": betti0_error(gt, pred),
    }
    if with_erl:
        from .erl import erl_from_masks
        erl, merged = erl_from_masks(gt, pred, gt_skel=gt_skel)
        if erl_ref is None:
            erl_ref, _ = erl_from_masks(gt, gt, gt_skel=gt_skel)
        out["erl"] = erl
        out["erl_frac"] = erl / erl_ref if erl_ref > 0 else 0.0
        out["merged_frac"] = merged
    return out
