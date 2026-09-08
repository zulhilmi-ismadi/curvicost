"""Reachable reference length -- the traceable-length cost of record.

Traceable length is measured on the REFERENCE skeleton, not on a re-skeletonised
perturbed mask: the count of reference-skeleton voxels that lie in the connected
component of the perturbed mask containing the reference root. Re-skeletonising
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

Root fallback (v7, 2026-09-08). Review panel v2's regenerated operator gallery
showed a 3D-tree truncation case at Dice 0.87 with reachable length 0.00: the
root of a single-component trace sat on the terminal branch the operator
erased, the v6 code looked for any mask voxel within a 3-voxel box, found none,
and counted the whole component as lost (15 of 504 3D-tree and 5 of 1,092
2D-tree truncate cases). The root is a bookkeeping choice, not a biological
one, so losing the root voxel must not lose the component. When the root voxel
is not foreground in the perturbed mask, the component is now traced from the
nearest SURVIVING reference-skeleton voxel of that same reference component
(the analogue of the pinned source, which attaches to the nearest graph node).
A component is lost only when none of its reference-skeleton voxels survives.
Whenever the root voxel survives the value is identical to v6.

The counting is vectorised over skeleton voxels (a pair histogram of reference
label x perturbed label) rather than a per-component pass over the volume; the
result is identical and the vascular blocks score in milliseconds rather than
minutes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _label(mask):
    return ndimage.label(mask, structure=np.ones((3,) * mask.ndim))


def _root_label(lab, p, ref_c, rl, pl, coords):
    """Perturbed-mask label reached from root voxel p of reference component ref_c.

    `rl`, `pl`, `coords`: reference label, perturbed label and coordinates of every
    reference-skeleton voxel. Returns 0 if the component has no surviving voxel."""
    inside = all(0 <= p[i] < lab.shape[i] for i in range(lab.ndim))
    l = int(lab[p]) if inside else 0
    if l:
        return l
    sel = (rl == ref_c) & (pl > 0)
    if not sel.any():
        return 0
    d = ((coords[sel] - np.asarray(p, dtype=float)) ** 2).sum(1)
    return int(pl[sel][int(d.argmin())])


def _skeleton_tables(mask, ref_skel, ref_lab, lab):
    idx = np.flatnonzero(np.asarray(ref_skel, dtype=bool))
    rl = np.asarray(ref_lab).ravel()[idx]
    pl = lab.ravel()[idx]
    coords = np.array(np.unravel_index(idx, mask.shape)).T.astype(float)
    return rl, pl, coords


def reach_count(mask, ref_skel, src_pos, ref_lab=None):
    """Reference-skeleton voxels inside the connected component of `mask` that
    contains the reference source position.

    With `ref_lab` (the labelled reference), a destroyed source voxel falls back to
    the nearest surviving reference-skeleton voxel of the source's own reference
    component. Without it, the v6 behaviour (nearest mask voxel within 6) is kept."""
    lab, n = _label(mask)
    if n == 0:
        return 0
    p = tuple(int(round(c)) for c in src_pos)
    inside = all(0 <= p[i] < mask.shape[i] for i in range(mask.ndim))
    l = int(lab[p]) if inside else 0
    if l == 0 and ref_lab is not None:
        ref_c = int(np.asarray(ref_lab)[p]) if inside else 0
        if ref_c == 0:
            # source position off the reference: nearest reference-skeleton voxel's component
            rl_all, _, coords_all = _skeleton_tables(mask, ref_skel, ref_lab, lab)
            if len(rl_all) == 0:
                return 0
            ref_c = int(rl_all[int(((coords_all - np.asarray(p, dtype=float)) ** 2).sum(1).argmin())])
        rl, pl, coords = _skeleton_tables(mask, ref_skel, ref_lab, lab)
        l = _root_label(lab, p, ref_c, rl, pl, coords)
        if l == 0:
            return 0
        return int((pl == l).sum())
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
    reference component's root voxel (or, if the root voxel was removed, the nearest
    surviving reference-skeleton voxel of the same component). Cannot exceed the
    reference total; a bridge between two reference fragments adds nothing (their
    voxels were already counted)."""
    lab, n = _label(mask)
    if n == 0:
        return 0
    rl, pl, coords = _skeleton_tables(mask, ref_skel, ref_lab, lab)
    if len(rl) == 0:
        return 0
    total = 0
    for c, root in ref_roots.items():
        p = tuple(int(round(x)) for x in root)
        l = _root_label(lab, p, c, rl, pl, coords)
        if l == 0:
            continue
        total += int(((rl == c) & (pl == l)).sum())
    return total


def ref_roots_for(gt, skel, edt):
    """Root voxel per reference component: the skeleton voxel of maximum distance-transform value."""
    ref_lab, n = _label(gt)
    roots = {}
    for c in range(1, n + 1):
        vox = np.argwhere(skel & (ref_lab == c))
        if len(vox) == 0:
            continue
        roots[c] = tuple(vox[np.argmax(edt[tuple(vox.T)])])
    return ref_lab, roots
