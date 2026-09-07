"""Reachable reference length -- the v6 traceable-length cost (data of record).

Traceable length is measured on the REFERENCE skeleton, not on a re-skeletonised
perturbed mask: the count of reference-skeleton voxels that lie in the connected
component of the perturbed mask containing the reference source. Re-skeletonising
the cut faces of a break inflated the graph-based value above 1 in half the
vascular break cases (review panel v1, 2026-09-07); counting reference voxels
through the perturbed mask's own connectivity is immune to that, and is bounded
by 1 from above by construction.

Two variants, both used by the study:
  reach_count        single pinned source (the reference's maximum-radius node)
  reach_count_multi  every reference component traced from its own root (the
                     skeleton voxel of maximum distance-transform value), so a
                     fragmented reference (a DIADEM layer-1 field of ~70 axons,
                     a boundary-clipped block) is fully covered. This is the
                     study's cost of record on the three length cells.

These functions were written in code/rescore_v6.py and moved here unchanged on
2026-09-08 so the public tool can vendor them; rescore_v6.py re-imports them.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def reach_count(mask, ref_skel, src_pos):
    """Reference-skeleton voxels inside the connected component of `mask` that
    contains the reference source position (nearest mask voxel if it was removed)."""
    lab, n = ndimage.label(mask, structure=np.ones((3,) * mask.ndim))
    if n == 0:
        return 0
    p = tuple(int(round(c)) for c in src_pos)
    l = lab[p] if all(0 <= p[i] < mask.shape[i] for i in range(mask.ndim)) else 0
    if l == 0:
        half = 6
        sl = tuple(slice(max(p[i] - half, 0), min(p[i] + half + 1, mask.shape[i])) for i in range(mask.ndim))
        sub = lab[sl]
        if sub.max() == 0:
            return 0
        idx = np.argwhere(sub > 0)
        off = np.array([s.start for s in sl])
        d = ((idx + off - np.array(p)) ** 2).sum(1)
        l = sub[tuple(idx[d.argmin()])]
    return int((ref_skel & (lab == l)).sum())


def reach_count_multi(mask, ref_skel, ref_lab, ref_roots):
    """Multi-root traceable length: for every reference component (label in ref_lab),
    its skeleton voxels that lie in the perturbed-mask component containing that
    reference component's root voxel. Cannot exceed the reference total; a bridge
    between two reference fragments adds nothing (their voxels were already counted)."""
    lab, n = ndimage.label(mask, structure=np.ones((3,) * mask.ndim))
    if n == 0:
        return 0
    total = 0
    for c, root in ref_roots.items():
        p = tuple(int(round(x)) for x in root)
        l = lab[p] if all(0 <= p[i] < mask.shape[i] for i in range(mask.ndim)) else 0
        if l == 0:
            lo = tuple(max(p[i] - 3, 0) for i in range(mask.ndim)); hi = tuple(min(p[i] + 4, mask.shape[i]) for i in range(mask.ndim))
            sub = lab[tuple(slice(a, b) for a, b in zip(lo, hi))]
            if sub.max() == 0:
                continue
            idx = np.argwhere(sub > 0); off = np.array([sl.start for sl in (slice(a, b) for a, b in zip(lo, hi))])
            d = ((idx + off - np.array(p)) ** 2).sum(1); l = sub[tuple(idx[d.argmin()])]
        total += int((ref_skel & (ref_lab == c) & (lab == l)).sum())
    return total


def ref_roots_for(gt, skel, edt):
    """Root voxel per reference component: the skeleton voxel of maximum distance-transform value."""
    ref_lab, n = ndimage.label(gt, structure=np.ones((3,) * gt.ndim))
    roots = {}
    for c in range(1, n + 1):
        vox = np.argwhere(skel & (ref_lab == c))
        if len(vox) == 0:
            continue
        roots[c] = tuple(vox[np.argmax(edt[tuple(vox.T)])])
    return ref_lab, roots
