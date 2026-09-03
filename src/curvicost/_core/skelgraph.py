"""Binary mask -> skeleton -> graph, for structures that are NOT trees.

Retinal vasculature has loops and several entry points, so the SWC Tree class
does not apply. Nodes are junctions and endpoints; an edge is the whole branch
between two of them, carrying its arc length and mean radius (from the distance
transform of the original mask, so radius is real thickness, not skeleton width).

This is the step DIADEM let us skip. Everything here is the part of the pipeline
that Gate P's retina half exists to test.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize

def _kernel(ndim):
    """All-ones neighbourhood with the centre removed (8-conn 2D, 26-conn 3D)."""
    k = np.ones((3,) * ndim, dtype=np.uint8)
    k[(1,) * ndim] = 0
    return k


def neighbour_count(skel):
    return ndimage.convolve(skel.astype(np.uint8), _kernel(skel.ndim), mode="constant")


def skeleton_to_graph(mask, skel=None, radius_map=None):
    """Build a branch graph. Returns (G, skel).

    Junction PIXELS are merged into junction NODES first. Skeletonisation emits
    a bifurcation as a blob of 4-10 adjacent degree>=3 pixels, not one pixel;
    treating each as its own node inflated a STARE graph 4x (427 "junctions"
    for 102 real ones, 73% of edges being 1-px links between them) and would
    make any Betti / component count meaningless.

    G nodes: integer ids, attr 'pos' (representative pixel), 'radius', 'kind'.
    G edges: attr 'length' (arc length px), 'radius' (mean), 'pixels'.
    """
    if skel is None:
        skel = skeletonize(mask)
    if radius_map is None:
        radius_map = ndimage.distance_transform_edt(mask)

    nb = neighbour_count(skel)
    deg = np.where(skel, nb, 0)
    junction_px = skel & (deg >= 3)
    endpoint_px = skel & (deg == 1)

    lab, n_clusters = ndimage.label(junction_px, structure=np.ones((3,) * skel.ndim))
    node_of = {}          # pixel -> node id
    G = nx.MultiGraph()

    # Group junction pixels by cluster in ONE pass. The obvious
    # `np.argwhere(lab == cid)` per cluster scans the whole volume each time --
    # O(clusters x voxels), which dominated every per-case graph rebuild on
    # 256^3 blocks (328/330 profiler samples sat in the uint32 compare).
    # argsort is stable, so coordinates stay in C order within each cluster and
    # `px` is byte-identical to the old expression, `centre` included.
    _coords = np.argwhere(lab)
    _cids = lab[tuple(_coords.T)]
    _order = np.argsort(_cids, kind="stable")
    _coords, _cids = _coords[_order], _cids[_order]
    _starts = np.searchsorted(_cids, np.arange(1, n_clusters + 2), side="left")

    for cid in range(1, n_clusters + 1):
        px = [tuple(p) for p in _coords[_starts[cid - 1]:_starts[cid]]]
        nid = cid - 1
        centre = px[len(px) // 2]
        G.add_node(nid, pos=centre, kind="junction",
                   radius=float(np.mean([radius_map[q] for q in px])), pixels=px)
        for q in px:
            node_of[q] = nid

    nid = n_clusters
    for q in (tuple(p) for p in np.argwhere(endpoint_px)):
        if q in node_of:
            continue
        G.add_node(nid, pos=q, kind="endpoint", radius=float(radius_map[q]), pixels=[q])
        node_of[q] = nid
        nid += 1

    seen = set()
    for px, src in list(node_of.items()):
        for step in _neighbours(px, skel):
            if node_of.get(step) == src:
                continue                      # inside the same junction blob
            key = (px, step)
            if key in seen:
                continue
            seen.add(key)
            path, prev, cur = [px, step], px, step
            while cur not in node_of:
                nxt = [q for q in _neighbours(cur, skel) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            seen.add((cur, prev))
            dst = node_of.get(cur)
            if dst is None:
                continue
            _add_edge(G, src, dst, path, radius_map)

    _add_orphan_cycles(G, skel, node_of, radius_map)
    return G, skel


_OFFSETS = {}


def _offsets(ndim):
    if ndim not in _OFFSETS:
        grid = np.indices((3,) * ndim).reshape(ndim, -1).T - 1
        _OFFSETS[ndim] = [tuple(o) for o in grid if any(o)]
    return _OFFSETS[ndim]


def _neighbours(p, skel):
    out = []
    for off in _offsets(skel.ndim):
        q = tuple(a + b for a, b in zip(p, off))
        if all(0 <= q[i] < skel.shape[i] for i in range(skel.ndim)) and skel[q]:
            out.append(q)
    return out


def _add_edge(G, src, dst, path, radius_map):
    d = np.diff(np.asarray(path, dtype=float), axis=0)
    length = float(np.sum(np.linalg.norm(d, axis=1))) if len(d) else 1.0
    rad = float(np.mean([radius_map[p] for p in path]))
    G.add_edge(src, dst, length=max(length, 1e-6), radius=max(rad, 0.5), pixels=path)


def _add_orphan_cycles(G, skel, node_of, radius_map):
    """Closed loops containing no junction never get visited above."""
    covered = set()
    for _, _, data in G.edges(data=True):
        covered.update(data["pixels"])
    remaining = {tuple(p) for p in np.argwhere(skel)} - covered - set(node_of)
    nid = max(G.nodes) + 1 if G.number_of_nodes() else 0
    while remaining:
        seed = remaining.pop()
        comp, stack = {seed}, [seed]
        while stack:
            u = stack.pop()
            for v in _neighbours(u, skel):
                if v in remaining:
                    remaining.discard(v); comp.add(v); stack.append(v)
        G.add_node(nid, pos=seed, kind="cycle",
                   radius=float(np.mean([radius_map[q] for q in comp])), pixels=sorted(comp))
        G.add_edge(nid, nid, length=float(len(comp)), pixels=sorted(comp),
                   radius=float(np.mean([radius_map[q] for q in comp])))
        nid += 1


def prune_spurs(G, min_length):
    """Drop degree-1 branches shorter than min_length. Returns (G, n_removed).

    Skeletonisation of a noisy mask sprouts short false branches; how many, and
    how sensitive the graph is to this threshold, is exactly what Gate P's
    retina half must quantify.
    """
    G = G.copy()
    removed = 0
    changed = True
    while changed:
        changed = False
        for n in [n for n in G.nodes if G.degree(n) == 1]:
            e = list(G.edges(n, data=True))
            if e and e[0][2]["length"] < min_length:
                G.remove_edge(e[0][0], e[0][1])
                removed += 1
                changed = True
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return G, removed


def graph_stats(G):
    lengths = [d["length"] for _, _, d in G.edges(data=True)]
    return dict(
        nodes=G.number_of_nodes(), edges=G.number_of_edges(),
        components=nx.number_connected_components(G),
        total_length=float(np.sum(lengths)) if lengths else 0.0,
        endpoints=sum(1 for n in G.nodes if G.degree(n) == 1),
        junctions=sum(1 for n in G.nodes if G.degree(n) >= 3),
    )
