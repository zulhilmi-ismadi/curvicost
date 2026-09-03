"""Functional cost on a general (looped, multi-entry) branch graph.

Retinal vasculature is not a tree, so the SWC cost functions do not apply:
there is no single root, and loops mean a break need not orphan anything.
Source is the largest-radius node -- a proxy for the optic disc / main trunk,
since STARE and DRIVE do not annotate the entry point. Documented as a
prototype choice, not a claim.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def pick_source(G):
    return max(G.nodes, key=lambda n: G.nodes[n].get("radius", 0.0))


def terminals(G):
    return [n for n in G.nodes if G.degree(n) == 1]


def reachable(G, source):
    return nx.node_connected_component(G, source) if source in G else set()


def perfused_fraction(G, source, reference_terminals):
    if not reference_terminals:
        return 1.0
    if source not in G:
        return 0.0
    reach = reachable(G, source)
    return float(sum(1 for t in reference_terminals if t in reach) / len(reference_terminals))


def traceable_length(G, source):
    if source not in G:
        return 0.0
    reach = reachable(G, source)
    return float(sum(d["length"] for u, v, d in G.edges(data=True)
                     if u in reach and v in reach))


def conductance(G, source, reference_terminals, eps=1e-9):
    """Parallel sum over reachable terminals of 1/(series resistance L/r^4).

    Uses the least-resistance path, so a loop can carry flow around a break --
    the behaviour that makes vessels different from trees.
    """
    if source not in G or not reference_terminals:
        return 0.0
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        r = max(d.get("radius", 0.5), eps)
        w = d["length"] / (r ** 4)
        if H.has_edge(u, v):
            H[u][v]["weight"] = min(H[u][v]["weight"], w)
        else:
            H.add_edge(u, v, weight=w)
    if source not in H:
        return 0.0
    dist = nx.single_source_dijkstra_path_length(H, source, weight="weight")
    return float(sum(1.0 / dist[t] for t in reference_terminals
                     if t in dist and dist[t] > 0))


def all_costs(G, source, reference_terminals):
    return dict(
        perfused_fraction=perfused_fraction(G, source, reference_terminals),
        traceable_length=traceable_length(G, source),
        conductance=conductance(G, source, reference_terminals),
    )


def break_edges(G, n_breaks, seed, mask=None, radius_map=None):
    """Remove n random branches; if a mask is given, cut a matching gap in it.

    The gap is a disk at the branch midpoint sized to local thickness -- a
    break, not a deleted branch, so the voxel cost stays small while the
    connectivity cost does not.
    """
    rng = np.random.default_rng(seed)
    edges = list(G.edges(keys=True, data=True))
    if not edges:
        return G.copy(), (None if mask is None else mask.copy()), {"n_broken": 0, "voxels": 0}
    pick = rng.choice(len(edges), size=min(n_breaks, len(edges)), replace=False)

    H = G.copy()
    out_mask = None if mask is None else mask.copy()
    voxels = 0
    for i in pick:
        u, v, k, d = edges[i]
        if H.has_edge(u, v, k):
            H.remove_edge(u, v, k)
        if out_mask is not None:
            px = d["pixels"]
            mid = px[len(px) // 2]
            r = d.get("radius", 1.0) + 1.0
            voxels += _punch(out_mask, mid, r)
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0])
    return H, out_mask, {"n_broken": len(pick), "voxels": voxels}


def _punch(mask, centre, r):
    """Clear a ball of radius r at `centre`. Works in 2D and 3D."""
    r = float(max(r, 1.0))
    lo = [max(int(np.floor(c - r)), 0) for c in centre]
    hi = [min(int(np.ceil(c + r)) + 1, s) for c, s in zip(centre, mask.shape)]
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    grids = np.ogrid[sl]
    ball = sum((g - c) ** 2 for g, c in zip(grids, centre)) <= r * r
    before = mask[sl].sum()
    mask[sl] &= ~ball
    return int(before - mask[sl].sum())


def conductance_fixed_sinks(G, source, sink_positions, match_radius=8.0, eps=1e-9):
    """Conductance to FIXED anatomical sink locations, not the graph's own leaves.

    Summing over the perturbed graph's own terminals rewards damage: every break
    manufactures two fresh endpoints, and endpoints near the source have low
    resistance, so conductance ROSE with severity (measured 1.06 -> 1.67 across
    the 1..16 break ladder -- backwards). Sinks must be the same physical points
    in every case, so a break that orphans a region makes those points
    unreachable and the conductance they carried is lost.

    sink_positions: array of reference terminal coordinates. Each is matched to
    the nearest node of G within `match_radius`; unmatched or unreachable sinks
    contribute zero.
    """
    if source not in G or len(sink_positions) == 0:
        return 0.0
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        r = max(d.get("radius", 0.5), eps)
        w = d["length"] / (r ** 4)
        if H.has_edge(u, v):
            H[u][v]["weight"] = min(H[u][v]["weight"], w)
        else:
            H.add_edge(u, v, weight=w)
    if source not in H:
        return 0.0

    dist = nx.single_source_dijkstra_path_length(H, source, weight="weight")

    # Candidates are TERMINALS only, and each may serve at most one sink.
    #
    # Matching a sink to the nearest node of any kind lets a destroyed terminal
    # re-attach to a thick interior junction. Edge weight is L/r^4, so a trunk
    # near the source has near-zero resistance and 1/dist explodes: on
    # BL6J-no1@790_3212_1223 two sinks landed on one degree-7 node of radius
    # 6.02 and contributed 2866 each, 98% of the total, turning a 10% break
    # into a 15.3x conductance GAIN. A sink is an anatomical perfusion
    # endpoint; if no endpoint survives near it, it is lost and contributes 0.
    cand = [n for n in G.nodes if "pos" in G.nodes[n] and G.degree(n) == 1]
    if not cand:
        return 0.0
    pts = np.array([G.nodes[n]["pos"] for n in cand], dtype=float)
    sps = np.asarray(sink_positions, dtype=float)

    # Greedy one-to-one assignment, closest pairs first, so a contested
    # terminal goes to the sink it actually represents.
    pairs = []
    for i, sp in enumerate(sps):
        d2 = ((pts - sp) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= match_radius ** 2:
            pairs.append((d2[j], i, j))
    pairs.sort()

    used_sink, used_node, total = set(), set(), 0.0
    for _, i, j in pairs:
        if i in used_sink or j in used_node:
            continue
        used_sink.add(i); used_node.add(j)
        n = cand[j]
        if n in dist and dist[n] > 0:
            total += 1.0 / dist[n]
    return float(total)
