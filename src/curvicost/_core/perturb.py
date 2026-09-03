"""Typed, seeded degradation operators.

Every operator changes exactly one property. Breaks act on the graph (an edge is
removed, so the change is exactly a removed connection); boundary noise acts on
the raster and must leave the graph alone. The unit tests in code/tests assert
that separation -- a leaky operator would invalidate the stratified analysis.
"""
from __future__ import annotations

import numpy as np

from .swcgraph import Tree


def break_edges(tree, severity=None, seed=0, protect_root=False, n_breaks=None):
    """Remove edges. Give either `n_breaks` (absolute) or `severity` (fraction).

    Absolute counts are the sane default for trees: the same FRACTION of edges
    means completely different damage depending on depth. On DIADEM, median
    root-to-leaf depth ranges 46 (OP_5) to 395 (OP_1), so a 1% break rate leaves
    most of a shallow arbor connected while severing a deep one entirely --
    every deep neuron floors at zero cost and the correlation signal dies.

    Detaching a child from its parent orphans the whole subtree below it, so a
    single break can cost far more arbor than its voxel count suggests. That
    asymmetry is the effect the study is built to measure.
    """
    rng = np.random.default_rng(seed)
    edges = tree.edges
    if n_breaks is None:
        if severity is None:
            raise ValueError("give n_breaks or severity")
        n_breaks = int(round(severity * len(edges)))
    n_break = int(n_breaks)
    out = tree.copy()
    if n_break == 0:
        return out, {"n_broken": 0, "broken_rows": [], "removed_length": 0.0}

    candidates = np.arange(len(edges))
    if protect_root:
        roots = set(tree.roots.tolist())
        candidates = np.array([i for i in candidates if int(edges[i, 1]) not in roots])
    chosen = rng.choice(candidates, size=min(n_break, len(candidates)), replace=False)

    lengths = tree.edge_lengths()
    for i in chosen:
        out.parent[int(edges[i, 0])] = -1  # detach child -> subtree orphaned
    return out, {
        "n_broken": len(chosen),
        "broken_rows": [int(edges[i, 0]) for i in chosen],
        "removed_length": float(lengths[chosen].sum()),
    }


def break_specific(tree, child_row):
    """Break one named edge -- used by the sanity anchors."""
    out = tree.copy()
    out.parent[child_row] = -1
    return out


def boundary_noise_matched(mask, n_voxels, seed):
    """Erode/dilate exactly n_voxels boundary voxels. Topology must survive.

    This is the control for a break: the same voxel budget spent on the object's
    surface instead of its connectivity.
    """
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    out = mask.copy()
    if n_voxels <= 0:
        return out

    eroded = ndimage.binary_erosion(out)
    shell = out & ~eroded                      # removable surface voxels
    idx = np.argwhere(shell)
    rng.shuffle(idx)

    removed = 0
    for pos in idx:
        if removed >= n_voxels:
            break
        t = tuple(pos)
        out[t] = False
        # a surface voxel whose removal splits the object is not boundary noise
        if _local_split(out, t):
            out[t] = True
            continue
        removed += 1
    return out


def _local_split(vol, centre, rad=1):
    """True if clearing `centre` disconnected its own 3x3(x3) neighbourhood."""
    from scipy import ndimage

    lo = [max(c - rad, 0) for c in centre]
    hi = [min(c + rad + 1, s) for c, s in zip(centre, vol.shape)]
    patch = vol[tuple(slice(l, h) for l, h in zip(lo, hi))]
    if patch.sum() == 0:
        return True
    _, n = ndimage.label(patch, structure=np.ones((3,) * vol.ndim))
    return n > 1


# ---------------------------------------------------------------------------
# Mask-level operators for the general (skeleton-graph) pipeline.
# Each changes exactly one property; code/tests/test_operator_leaks.py asserts it.
# ---------------------------------------------------------------------------

def _draw_tube(mask, a, b, radius):
    """Stamp a ball of `radius` along the segment a->b. Returns voxels added."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = max(int(np.ceil(np.linalg.norm(b - a) * 2)), 1)
    before = int(mask.sum())
    for t in np.linspace(0.0, 1.0, n + 1):
        c = a + (b - a) * t
        lo = [max(int(np.floor(x - radius)), 0) for x in c]
        hi = [min(int(np.ceil(x + radius)) + 1, s) for x, s in zip(c, mask.shape)]
        if any(l >= h for l, h in zip(lo, hi)):
            continue
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        grids = np.ogrid[sl]
        ball = sum((g - x) ** 2 for g, x in zip(grids, c)) <= radius * radius
        mask[sl] |= ball
    return int(mask.sum()) - before


def spurious_bridges(mask, G, n_bridges, seed, max_gap=12.0, min_graph_hops=6):
    """Add false connections between branches that are near in SPACE but far in
    the GRAPH. Adds connectivity; must never remove any.

    Candidate pairs are endpoint nodes within `max_gap` voxels whose graph
    distance is at least `min_graph_hops` (or infinite, i.e. different
    components) -- otherwise a "bridge" would merely thicken an existing link.
    """
    import networkx as nx

    rng = np.random.default_rng(seed)
    out = mask.copy()
    ends = [n for n in G.nodes if G.degree(n) == 1]
    if len(ends) < 2:
        return out, {"n_bridges": 0, "voxels": 0}

    pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in ends}
    cands = []
    for i, u in enumerate(ends):
        for v in ends[i + 1:]:
            d = float(np.linalg.norm(pos[u] - pos[v]))
            if d > max_gap or d < 1.0:
                continue
            try:
                hops = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                hops = np.inf
            if hops >= min_graph_hops:
                cands.append((u, v, d))
    if not cands:
        return out, {"n_bridges": 0, "voxels": 0}

    pick = rng.choice(len(cands), size=min(n_bridges, len(cands)), replace=False)
    added = 0
    for i in pick:
        u, v, _ = cands[int(i)]
        r = max(min(G.nodes[u].get("radius", 1.0), G.nodes[v].get("radius", 1.0)), 1.0)
        added += _draw_tube(out, pos[u], pos[v], r)
    return out, {"n_bridges": len(pick), "voxels": added}


def ownership_map(mask, skel):
    """For every mask voxel, the index of its nearest skeleton voxel.

    Needed so an operator can erase only the tissue belonging to ONE branch.
    Erasing a ball around a branch pixel also clears any neighbouring vessel
    passing within that radius -- in dense retina that severed bystanders and
    added 16 spurious components to a truncation run.
    """
    from scipy import ndimage

    idx = np.argwhere(skel)
    _, inds = ndimage.distance_transform_edt(~skel, return_indices=True)
    flat = np.ravel_multi_index([inds[i] for i in range(mask.ndim)], mask.shape)
    lookup = -np.ones(int(np.prod(mask.shape)), dtype=np.int64)
    lookup[np.ravel_multi_index([idx[:, i] for i in range(mask.ndim)], mask.shape)] = \
        np.arange(len(idx))
    return lookup[flat].reshape(mask.shape), {tuple(p): i for i, p in enumerate(idx)}


def endpoint_truncation(mask, G, n_branches, seed, fraction=0.5, skel=None, owner=None):
    """Erase the distal `fraction` of randomly chosen terminal branches.

    Shortening a free end cannot disconnect anything, so component count must be
    unchanged -- that is the leak test. Erasure is restricted to voxels the cut
    pixels OWN (nearest-skeleton-voxel), so a neighbouring vessel running close
    by is left intact. Mimics an annotator or model that stops tracing where a
    vessel fades.
    """
    from skimage.morphology import skeletonize

    rng = np.random.default_rng(seed)
    out = mask.copy()
    term_edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
                  if (G.degree(u) == 1) != (G.degree(v) == 1)]
    if not term_edges:
        return out, {"n_truncated": 0, "voxels": 0}

    if skel is None:
        skel = skeletonize(mask)
    if owner is None:
        owner = ownership_map(mask, skel)
    owner_idx, px_to_i = owner

    from scipy import ndimage
    struct = np.ones((3,) * mask.ndim)
    _, base_comps = ndimage.label(mask, structure=struct)

    pick = rng.choice(len(term_edges), size=min(n_branches, len(term_edges)), replace=False)
    applied, removed, skipped = 0, 0, 0
    for i in pick:
        u, v, k, d = term_edges[int(i)]
        px = list(d["pixels"])
        leaf_at_start = G.degree(u) == 1
        n_cut = max(int(len(px) * fraction), 1)
        cut = px[:n_cut] if leaf_at_start else px[-n_cut:]
        ids = {px_to_i[tuple(q)] for q in cut if tuple(q) in px_to_i}
        if not ids:
            continue
        doomed = out & np.isin(owner_idx, list(ids))
        if not doomed.any():
            continue
        trial = out & ~doomed
        # Two truncations meeting at one junction can strand the stub between
        # them. Applying branch-by-branch and rejecting any cut that raises the
        # component count keeps "truncation never disconnects" exactly true.
        _, comps = ndimage.label(trial, structure=struct)
        if comps > base_comps:
            skipped += 1
            continue
        out = trial
        removed += int(doomed.sum())
        applied += 1
    return out, {"n_truncated": applied, "voxels": removed, "skipped": skipped}


def _erase_ball(mask, centre, radius):
    radius = float(max(radius, 1.0))
    lo = [max(int(np.floor(c - radius)), 0) for c in centre]
    hi = [min(int(np.ceil(c + radius)) + 1, s) for c, s in zip(centre, mask.shape)]
    if any(l >= h for l, h in zip(lo, hi)):
        return 0
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    grids = np.ogrid[sl]
    ball = sum((g - c) ** 2 for g, c in zip(grids, centre)) <= radius * radius
    before = int(mask[sl].sum())
    mask[sl] &= ~ball
    return before - int(mask[sl].sum())


def radius_bias(mask, scale, skel=None, radius_map=None, min_radius=1.0,
                preserve_topology=True):
    """Systematically thicken/thin the structure, preserving topology.

    Redraws from the SKELETON with scaled radii rather than dilating/eroding the
    mask: erosion snaps thin branches (changing Betti-0) and dilation fuses
    neighbours (changing Betti-1), which would make this operator leak into the
    topology channel. Radii are clamped at `min_radius` so nothing vanishes.
    STUDY-PLAN 4.1 requires Betti numbers to survive this operator untouched.

    NOTE the reference for this arm is `radius_bias(mask, 1.0)`, NOT the original
    mask: the union-of-balls reconstruction is systematically fatter than the
    input (+7856 voxels on a STARE image at scale 1.0) and can fuse a nearby
    component. Comparing scaled reconstructions against the scale-1.0
    reconstruction keeps the operator's only variable the radius.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    if skel is None:
        skel = skeletonize(mask)
    if radius_map is None:
        radius_map = ndimage.distance_transform_edt(mask)

    # Thickening fuses separate fragments that happen to lie close together --
    # a real geometric fact, not a bug, but it would leak the radius operator
    # into the topology channel (measured: Betti-0 11->10 at scale 1.3 on
    # STARE im0004). With preserve_topology, a voxel is only painted if its
    # NEAREST skeleton voxel belongs to the same connected component as the
    # voxel doing the painting, so components cannot merge by construction.
    comp_of = None
    if preserve_topology:
        lab, _ = ndimage.label(skel, structure=np.ones((3,) * mask.ndim))
        owner_idx, px_to_i = ownership_map(mask | _grow_room(mask, scale, radius_map), skel)
        idx = np.argwhere(skel)
        comp_of = lab[tuple(idx.T)]
        owner_comp = np.where(owner_idx >= 0, comp_of[np.clip(owner_idx, 0, None)], 0)

    out = np.zeros_like(mask)
    for j, p in enumerate(np.argwhere(skel)):
        p = tuple(p)
        r = max(float(radius_map[p]) * scale, min_radius)
        lo = [max(int(np.floor(c - r)), 0) for c in p]
        hi = [min(int(np.ceil(c + r)) + 1, s) for c, s in zip(p, mask.shape)]
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        grids = np.ogrid[sl]
        ball = sum((g - c) ** 2 for g, c in zip(grids, p)) <= r * r
        if comp_of is not None:
            ball &= (owner_comp[sl] == comp_of[j])
        out[sl] |= ball
    return out, {"scale": scale, "voxels": int(out.sum()) - int(mask.sum())}


def _grow_room(mask, scale, radius_map):
    """Dilation envelope the scaled structure could occupy (for ownership)."""
    from scipy import ndimage as ndi
    extra = int(np.ceil(max(float(radius_map.max()) * max(scale - 1.0, 0.0), 1.0)))
    return ndi.binary_dilation(mask, iterations=max(extra, 1))


# ---------------------------------------------------------------------------
# Severity scaling
# ---------------------------------------------------------------------------
# EXECUTION-PLAN.md:86 specifies severity as a FRACTION of the available
# targets ("breaks: 1%, 2%, 5%, 10%, 20% of edges"). The runners originally
# used an absolute count ladder [1,2,4,8,16], which is a heavy perturbation on
# a ~200-edge retina graph and a no-op on a ~7000-edge VesSAP block: at top
# severity it removed 25-44% of traceable length in the 2D/tree cells but only
# 0.2% in VesSAP, leaving that cell with no signal to correlate. See
# results/analysis/GATE-R.md.
# Extended 2026-09-02 after item 12: the original top severity (0.20) left every
# perturbation milder than STARE's inter-observer disagreement (harshest
# synthetic Dice 0.805 vs observer 0.740). 0.35/0.50 verified leak-free on the
# operator invariants (6 images x 3 seeds) before adoption.
FRAC_LADDER = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


def target_population(G, operator):
    """Number of targets `operator` can act on in G -- the denominator severity
    is a fraction of. Each operator is scaled by the population it draws from,
    so "20%" means the same thing across datasets and across operators."""
    if operator == "break":                 # break_edges picks from G.edges
        return G.number_of_edges()
    if operator == "bridge":                # spurious_bridges pairs degree-1 nodes
        return sum(1 for n in G.nodes if G.degree(n) == 1)
    if operator == "truncate":              # endpoint_truncation picks terminal branches
        return sum(1 for u, v, k, d in G.edges(keys=True, data=True)
                   if (G.degree(u) == 1) != (G.degree(v) == 1))
    raise ValueError(f"unknown operator: {operator}")


def n_from_frac(frac, population):
    """Absolute count for a fractional severity; always at least 1 target."""
    return max(1, int(round(frac * population)))
