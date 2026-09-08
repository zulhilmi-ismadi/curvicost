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


def break_edges(G, n_breaks, seed, mask=None, radius_map=None, sever=True, max_grow=6):
    """Remove n random branches; if a mask is given, cut a matching gap in it.

    The gap is a ball at the branch midpoint sized to local thickness -- a
    break, not a deleted branch, so the voxel cost stays small while the
    connectivity cost does not.

    Thickness is taken from the mask's own distance transform at the midpoint
    (never from an external radius map: the VesSAP radius file is zero on most
    skeleton voxels, and sizing the ball from it left ~80% of 3D cuts as
    constrictions rather than breaks -- found 2026-09-07 by the review-panel
    diagnostic, code/diag_flip_traceable.py). With `sever=True` every cut is
    verified in a local crop: the ball grows by one voxel at a time (up to
    `max_grow`) until the two sides of the branch fall into different
    connected components. The returned info counts cuts that could not be
    severed within the cap (`unsevered`), so a leak is reported, not silent.
    In 2D the first attempt equals the pre-fix ball, so 2D results are
    unchanged wherever the old ball already severed.
    """
    rng = np.random.default_rng(seed)
    edges = list(G.edges(keys=True, data=True))
    if not edges:
        return G.copy(), (None if mask is None else mask.copy()), {"n_broken": 0, "voxels": 0, "unsevered": 0}
    pick = rng.choice(len(edges), size=min(n_breaks, len(edges)), replace=False)

    H = G.copy()
    out_mask = None if mask is None else mask.copy()
    voxels, unsevered = 0, 0
    edt = None
    if out_mask is not None:
        from scipy import ndimage
        edt = ndimage.distance_transform_edt(mask)
    for i in pick:
        u, v, k, d = edges[i]
        if H.has_edge(u, v, k):
            H.remove_edge(u, v, k)
        if out_mask is not None:
            px = d["pixels"]
            mid = px[len(px) // 2]
            r = max(float(edt[tuple(mid)]), float(d.get("radius", 1.0))) + 1.0
            if sever:
                v_, ok = _punch_severing(out_mask, mid, r, px, max_grow)
                voxels += v_
                unsevered += (not ok)
            else:
                voxels += _punch(out_mask, mid, r)
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0])
    return H, out_mask, {"n_broken": len(pick), "voxels": voxels, "unsevered": unsevered}


def _punch_severing(mask, centre, r0, pixels, max_grow):
    """Punch a ball of radius r0 at `centre`; if the branch's two sides are still
    connected inside a local crop, grow the ball one voxel at a time (at most
    `max_grow` times). Returns (voxels removed, severed?).

    The two sides are the first branch pixels outside the current ball on each
    end of the branch. The crop is a box of half-size r+4 around the centre, so
    a genuine detour around a capillary loop (perimeter far larger than the
    crop) does not read as "unsevered"."""
    from scipy import ndimage
    px = np.asarray(pixels, dtype=int)
    c = np.asarray(centre, dtype=float)
    dist = np.sqrt(((px - c) ** 2).sum(axis=1))
    before = int(mask.sum()) if mask.ndim == 2 else None   # 2D masks are small; 3D counted per crop
    r = float(max(r0, 1.0))
    removed_total = 0
    for _ in range(max_grow + 1):
        removed_total += _punch(mask, centre, r)
        outside = np.where(dist > r + 0.5)[0]
        if len(outside) == 0:
            return removed_total, True          # whole branch inside the ball: severed by construction
        a, b = px[outside[0]], px[outside[-1]]
        if outside[0] == outside[-1] or (dist[outside[0]] <= r + 0.5) or (dist[outside[-1]] <= r + 0.5):
            return removed_total, True
        half = int(np.ceil(r)) + 4
        lo = [max(int(cc - half), 0) for cc in centre]
        hi = [min(int(cc + half) + 1, s) for cc, s in zip(centre, mask.shape)]
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        crop = mask[sl]
        ia = tuple(int(q - l) for q, l in zip(a, lo)); ib = tuple(int(q - l) for q, l in zip(b, lo))
        inside = all(0 <= ia[j] < crop.shape[j] for j in range(crop.ndim)) and \
                 all(0 <= ib[j] < crop.shape[j] for j in range(crop.ndim))
        if not inside:
            return removed_total, True          # ends beyond the crop: treat as severed (cannot verify)
        lab, _ = ndimage.label(crop, structure=np.ones((3,) * crop.ndim))
        if lab[ia] == 0 or lab[ib] == 0 or lab[ia] != lab[ib]:
            return removed_total, True
        r += 1.0
    return removed_total, False


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

def conductance_kirchhoff(G, source, sink_positions, mask=None, radius_map=None, match_radius=3.0, eps=1e-9, extra_sources=(),
                          terminal_resistance=0.0):
    """Effective hydraulic conductance from the source to the reference sinks by a
    Kirchhoff (Laplacian) solve on the skeleton graph: unit pressure at the source,
    zero pressure at every surviving sink, edge conductance r^4 / L (parallel edges
    add), total outflow from the source returned. Unlike the least-resistance-path
    sum (`conductance_fixed_sinks`), this is a resistor-network conductance: adding
    an edge can never lower it (Rayleigh monotonicity) and parallel paths carry
    flow, which is the physics a looped network needs.

    Sink survival is decided by the MASK and by skeleton proximity, never by node
    degree: a reference terminal location still contributes if its voxel (or a
    26-neighbour) is foreground in the perturbed mask and a skeleton pixel of the
    perturbed graph lies within `match_radius`. The sink attaches to the nearest
    skeleton pixel: a node, or an interior pixel of an edge, in which case the edge
    is split there. A terminal that a bridge turned into a through-point keeps its
    sink (it still exists); a terminal a break punched away loses it. This replaces
    the terminal-node-only rule, under which bridges read as a 41% conductance
    loss (review panel v1, 2026-09-07).

    `terminal_resistance` is the boundary condition on the sinks (review panel v2,
    R7). At the default 0.0 every surviving sink is an ideal ground held at zero
    pressure, which is the model of record. Given a positive value R_t, the sinks
    are instead free nodes that drain to a single common ground through a lumped
    resistance R_t each -- the downstream bed the imaged network empties into. The
    two limits bracket the physics: R_t -> 0 recovers the ideal ground, and a large
    R_t makes the terminal beds, not the imaged vessels, the flow-limiting element,
    so the cost stops discriminating between segmentations. Callers set R_t as a
    multiple of the unit's own reference conductance (see code/run_bc_sensitivity.py)
    so that the ladder is scale-free and fixed per unit rather than tuned per case.
    """
    if source not in G or len(sink_positions) == 0:
        return 0.0
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import cKDTree
    sources = [source] + [x for x in extra_sources if x in G and x != source]   # multi-root: every source at unit pressure
    sps = np.asarray(sink_positions, dtype=float)
    # candidate attachment points: every skeleton pixel of every edge, plus nodes
    cand_pts, cand_ref = [], []            # ref = ("n", node) or ("e", (u, v, k), index)
    for n in G.nodes:
        if "pos" in G.nodes[n]:
            cand_pts.append(G.nodes[n]["pos"]); cand_ref.append(("n", n, None))
    for u, v, k, d in G.edges(keys=True, data=True):
        px = d.get("pixels") or []
        for i, q in enumerate(px):
            # end pixels of an edge ARE its nodes: attach there, never split at an end
            if i == 0:
                cand_pts.append(q); cand_ref.append(("n", u, None))
            elif i == len(px) - 1:
                cand_pts.append(q); cand_ref.append(("n", v, None))
            else:
                cand_pts.append(q); cand_ref.append(("e", (u, v, k), i))
    if not cand_pts:
        return 0.0
    tree = cKDTree(np.asarray(cand_pts, dtype=float))
    alive = np.ones(len(sps), dtype=bool)
    if mask is not None:
        # a sink survives only if its location is still foreground AND lies in the
        # same mask component as the source: a severed terminal fragment is not
        # perfused, and it must not re-attach to a neighbouring vessel (the 15x
        # "gain" artefact of Supplementary Fig. 4).
        from scipy import ndimage
        lab, _ = ndimage.label(mask, structure=np.ones((3,) * mask.ndim))
        src_labs = set()
        for s_node in sources:
            sq = tuple(int(round(c)) for c in G.nodes[s_node]["pos"])
            l_ = lab[sq] if all(0 <= sq[i] < mask.shape[i] for i in range(mask.ndim)) else 0
            if l_ == 0:
                lo = tuple(max(c - 2, 0) for c in sq); hi = tuple(min(c + 3, dd) for c, dd in zip(sq, mask.shape))
                sub = lab[tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))]
                l_ = int(sub.max()) if sub.size else 0
            if l_:
                src_labs.add(int(l_))
        for i, s in enumerate(sps):
            q = tuple(int(round(c)) for c in s)
            lo = tuple(max(c - 1, 0) for c in q); hi = tuple(min(c + 2, dd) for c, dd in zip(q, mask.shape))
            sl = tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))
            alive[i] = bool(np.isin(lab[sl], list(src_labs)).any()) if (src_labs and all(a_ < b_ for a_, b_ in zip(lo, hi))) else False
    dist, j = tree.query(sps, distance_upper_bound=match_radius)
    # greedy one-to-one on attachment points
    order = np.argsort(dist)
    used, attach = set(), []
    for i in order:
        if not alive[i] or not np.isfinite(dist[i]):
            continue
        key = cand_ref[j[i]][:2] if cand_ref[j[i]][0] == "n" else (cand_ref[j[i]][1], cand_ref[j[i]][2])
        if key in used:
            continue
        used.add(key); attach.append(cand_ref[j[i]])
    if not attach:
        return 0.0
    # build the resistor network, splitting edges where sinks attach inside them
    splits = {}
    for ref in attach:
        if ref[0] == "e":
            splits.setdefault(ref[1], []).append(ref[2])
    nodes = list(G.nodes); idx = {n: k for k, n in enumerate(nodes)}
    sink_ids = set(idx[ref[1]] for ref in attach if ref[0] == "n")
    rows, cols, vals = [], [], []
    def add(a_, b_, g):
        rows.extend([a_, b_]); cols.extend([b_, a_]); vals.extend([g, g])

    def seg_conductance(px, i0, i1, r_mean, L_total):
        """1 / sum over pixel steps of dl / r(pixel)^4, radius read per pixel from the
        distance transform so a thin stretch (a bridge) does not contaminate the
        resistance of the thick stretch it was merged with. Falls back to the
        edge's mean radius when no radius map is given."""
        if radius_map is None or len(px) < 2:
            frac = max(i1 - i0, 0.5) / max(len(px) - 1, 1)
            return (max(r_mean, eps) ** 4) / max(L_total * frac, eps)
        R = 0.0
        for a_, b_ in zip(px[i0:i1], px[i0 + 1:i1 + 1]):
            dl = float(np.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a_, b_))))
            r = max(float(radius_map[tuple(b_)]), 0.5)
            R += dl / (r ** 4)
        return 1.0 / max(R, eps)

    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            continue
        r = max(d.get("radius", 0.5), eps); L = max(float(d["length"]), eps)
        px = d.get("pixels") or []
        cuts = sorted(i for i in set(splits.get((u, v, k), [])) if 0 < i < len(px) - 1)
        if not cuts or len(px) < 3:
            add(idx[u], idx[v], seg_conductance(px, 0, len(px) - 1, r, L) if len(px) >= 2 else (r ** 4) / L)
            continue
        bounds = [0] + cuts + [len(px) - 1]
        prev = idx[u]
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            if b1 == len(px) - 1:
                nxt = idx[v]
            else:
                nxt = len(nodes); nodes.append(("split", u, v, k, b1)); sink_ids.add(nxt)
            add(prev, nxt, seg_conductance(px, b0, b1, r, L)); prev = nxt
    n = len(nodes)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # restrict to the source's component
    ncomp, lab = sp.csgraph.connected_components(A, directed=False)
    src_idx = [idx[x] for x in sources]
    src_comp = set(int(lab[x]) for x in src_idx)
    keep = np.where(np.isin(lab, list(src_comp)))[0]
    sinks = [t for t in sink_ids if int(lab[t]) in src_comp and t not in src_idx]
    if not sinks:
        return 0.0
    gnd = None
    if terminal_resistance and float(terminal_resistance) > 0:
        # lumped terminal beds: each surviving sink drains to ONE common ground through
        # R_t, instead of being an ideal ground itself. The component restriction above
        # already decided which sinks survive, so the ground touches only those.
        g_t = 1.0 / float(terminal_resistance)
        gnd = n
        for t in sinks:
            add(t, gnd, g_t)
        n += 1
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        keep = np.append(keep, gnd)
    deg = np.asarray(A.sum(axis=1)).ravel()
    Lap = (sp.diags(deg) - A).tocsr()
    fixed = np.zeros(n, dtype=bool); pval = np.zeros(n)
    for x in src_idx:
        fixed[x] = True; pval[x] = 1.0
    if gnd is None:
        for t in sinks:
            fixed[t] = True
    else:
        fixed[gnd] = True                      # the sinks are now free nodes
    free = np.array([k for k in keep if not fixed[k]], dtype=int); fix = np.array([k for k in keep if fixed[k]], dtype=int)
    pres = pval.copy()
    if len(free):
        Lff = Lap[free][:, free].tocsc(); Lfb = Lap[free][:, fix]
        rhs = -Lfb @ pval[fix]
        try:
            pres[free] = spla.spsolve(Lff, rhs)
        except Exception:
            pres[free] = spla.lsqr(Lff, rhs)[0]
    out = 0.0
    for x in src_idx:
        row = A.getrow(x)
        out += float(sum(row.data[k] * (1.0 - pres[row.indices[k]]) for k in range(len(row.data))))
    return max(out, 0.0)


def component_sources(G):
    """One source per connected component of G: its maximum-radius node. Used for the
    multi-root costs, where every reference fragment (a separate axon in a DIADEM
    layer-1 field, a boundary-clipped vessel in a block) is traced from its own root."""
    return [max(cc, key=lambda n: G.nodes[n].get("radius", 0.0)) for cc in nx.connected_components(G)]
