"""Expected Run Length (ERL) and a DIADEM-style consequence-weighted score.

ERL, from the FFN definition (Januszewski et al. 2018, Nat Methods): pick a
random point on the ground-truth skeleton, follow the structure, and record how
far you travel before hitting an error. A MERGE is catastrophic -- any segment
that fuses two ground-truth objects contributes ZERO run length, not a reduced
one, which is what makes ERL different in kind from a length-weighted score.

Gate A1 makes the DIADEM comparison mandatory: DIADEM weights a missed node by
the arbor it orphans, i.e. it is the one pre-existing consequence-weighted
curvilinear metric. `diadem_like_score` reproduces that weighting on our graphs
(degree/subtree weighted node agreement) so the baseline exists even where
PyNeval's SWC-only interface does not apply (retina and VesSAP are not trees).
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def expected_run_length(gt_components, pred_labels_at, weights=None):
    """ERL over ground-truth objects.

    gt_components: list of lists of skeleton nodes, one per ground-truth object.
    pred_labels_at: callable node -> predicted segment id (None/0 = background).
    weights: optional node -> length contribution (default 1 per node).

    Returns (erl, merged_fraction). A predicted segment covering nodes from more
    than one ground-truth object is a merge; every object it touches scores 0.
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
    """Consequence-weighted node agreement, in the spirit of the DIADEM metric.

    Each reference node is weighted by the amount of arbor it would orphan if
    lost (subtree length below it, or its edge length for non-tree graphs), then
    matched to the nearest predicted node within `match_radius`. Score is the
    weighted fraction matched -- so losing a trunk node costs far more than
    losing a leaf, which is precisely the property [gillette-2011] introduced
    and that plain overlap metrics lack.
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
