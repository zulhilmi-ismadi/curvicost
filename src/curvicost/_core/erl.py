"""Expected run length (ERL) and a DIADEM-like length-weighted node score.

ERL follows the flood-filling-network definition (Januszewski et al. 2018, Nat
Methods): pick a random point on the ground-truth skeleton, follow the structure,
and record how far you travel before hitting an error; a MERGE is catastrophic.
`expected_run_length` implements this per ground-truth component and returns a
*component-normalised* ERL: per component i, sum_j r_ij^2 / L_i (the expected
error-free run of a random point in that component), summed over components, so
that the fraction against the reference is the length-weighted mean over
components of each component's own normalised expected run length. Any
ground-truth component touched by a merged segment contributes zero in full.
For a single-component reference this coincides with the FFN random-point ERL;
for a multi-component reference it differs from it (the FFN quantity is
L^2-weighted, and zeroes only the merged segment rather than every run in the
touched component). See the function docstring.

`diadem_like_score` is a DIADEM-*like* score, not the DIADEM metric: it weights
each reference node by the summed length of its incident edges and credits it if
any predicted node lies within `match_radius`. It is NOT rooted and computes no
subtree or orphaned-arbor weight, so it does not reproduce the consequence
weighting of Gillette et al. 2011; it exists so that a matching-style baseline
is available on graphs where PyNeval's SWC-only interface does not apply (retina
and VesSAP are not trees).
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def expected_run_length(gt_components, pred_labels_at, weights=None):
    """Component-normalised ERL over ground-truth objects.

    gt_components: list of lists of skeleton nodes, one per ground-truth object.
    pred_labels_at: callable node -> predicted segment id (None/0 = background).
    weights: optional node -> length contribution (default 1 per node).

    Returns (erl, merged_fraction).

    What is computed. For ground-truth component i of length L_i (sum of node
    weights), let r_ij be the length of component i that lies in predicted
    segment j. The component's expected error-free run for a random point on
    it is sum_j r_ij^2 / L_i, and `erl` is that quantity SUMMED over components.
    A predicted segment covering nodes from more than one ground-truth object
    is a merge; every object it touches contributes zero in full (not only the
    merged runs). `merged_fraction` is the length of those objects over the
    total length.

    How to read it. Dividing `erl` by the same quantity computed on the intact
    reference, as `erl_frac` does, gives the length-weighted mean over
    components of each component's own normalised expected run length -- a
    "component-normalised ERL". For a single-component reference this is
    exactly the flood-filling-network random-point ERL (Januszewski et al.
    2018). For a multi-component reference it is not: the FFN definition picks
    the random point over the WHOLE skeleton, which weights components by L_i^2
    rather than L_i, and zeroes only the merged segment's run rather than every
    run in the touched component. The component-normalised form is used
    deliberately so that one long trunk does not dominate a fragmented
    reference; the difference is nil on single-component references.

    Note on the last line: `weighted * total_len / total_len` is a no-op kept
    from an earlier L^2-weighted draft; the value returned is `weighted`, i.e.
    the sum over components of sum_j r_ij^2 / L_i.
    """
    w = (lambda n: 1.0) if weights is None else weights

    seg_to_objs = {}
    for oi, comp in enumerate(gt_components):
        for n in comp:
            p = pred_labels_at(n)
            if p:
                seg_to_objs.setdefault(p, set()).add(oi)
    merged_segs = {p for p, objs in seg_to_objs.items() if len(objs) > 1}

    total_len, weighted = 0.0, 0.0
    merged_len = 0.0
    for comp in gt_components:
        runs = {}
        obj_len = sum(w(n) for n in comp)
        total_len += obj_len
        bad = False
        for n in comp:
            p = pred_labels_at(n)
            if not p:
                continue
            if p in merged_segs:
                bad = True
                continue
            runs[p] = runs.get(p, 0.0) + w(n)
        if bad:
            merged_len += obj_len
            continue                      # merge => this object contributes zero
        if obj_len > 0 and runs:
            # expected run length for a random point in this object
            weighted += sum(r * r for r in runs.values()) / obj_len
    if total_len == 0:
        return 0.0, 0.0
    return float(weighted * total_len / total_len), float(merged_len / total_len)


def diadem_like_score(G_ref, G_pred, match_radius=6.0):
    """Length-weighted node agreement, in the spirit of (but not) the DIADEM metric.

    Each reference node is weighted by the summed length of its incident edges
    (a degree-1 tip carries one edge length, a junction carries the sum of its
    branches) and is credited if ANY predicted node lies within `match_radius`;
    predicted nodes may be reused, so this is not a one-to-one matching. The
    score is the weighted fraction of reference nodes credited.

    It is NOT rooted and computes no subtree or orphaned-arbor weight: losing a
    trunk node costs its own incident edge length, not the arbor below it. That
    is the property the DIADEM metric (Gillette et al. 2011) has and this score
    does not; it is a matching-style baseline that runs on non-tree graphs, no
    more.
    """
    if G_ref.number_of_nodes() == 0:
        return 1.0
    ref_nodes = [n for n in G_ref.nodes if "pos" in G_ref.nodes[n]]
    if not ref_nodes:
        return 1.0
    weight = {}
    for n in ref_nodes:
        weight[n] = float(sum(d.get("length", 1.0) for _, _, d in G_ref.edges(n, data=True)))
    tot = sum(weight.values())
    if tot <= 0:
        return 1.0

    pred_nodes = [n for n in G_pred.nodes if "pos" in G_pred.nodes[n]]
    if not pred_nodes:
        return 0.0
    pts = np.array([G_pred.nodes[n]["pos"] for n in pred_nodes], dtype=float)

    got = 0.0
    for n in ref_nodes:
        p = np.asarray(G_ref.nodes[n]["pos"], dtype=float)
        d2 = ((pts - p) ** 2).sum(axis=1)
        if d2.min() <= match_radius ** 2:
            got += weight[n]
    return float(got / tot)


def erl_from_masks(gt_mask, pred_mask, gt_skel=None):
    """ERL for a segmentation pair, using mask components as 'segments'.

    Ground-truth objects are the connected components of the GT skeleton; the
    predicted segment at a node is the connected-component label of the
    perturbed mask at that voxel. Node weight is one skeleton voxel, so ERL is
    in skeleton-voxel units and comparable across cases of the same image.

    Returns (erl, merged_fraction).
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    if gt_skel is None:
        gt_skel = skeletonize(gt_mask)
    struct = np.ones((3,) * gt_mask.ndim)

    gt_lab, _ = ndimage.label(gt_skel, structure=struct)
    pred_lab, _ = ndimage.label(pred_mask, structure=struct)

    comps = {}
    for p in np.argwhere(gt_skel):
        t = tuple(p)
        comps.setdefault(int(gt_lab[t]), []).append(t)
    gt_components = list(comps.values())
    if not gt_components:
        return 0.0, 0.0

    def label_at(node):
        v = int(pred_lab[node])
        return v if v > 0 else None

    return expected_run_length(gt_components, label_at)
