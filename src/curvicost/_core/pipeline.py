"""One honest path from a binary mask to its functional cost.

The reference and every perturbed case go through the SAME steps: mask ->
skeleton -> graph -> cost. Nothing is carried over from the reference graph,
because in real use you are handed a segmentation, not a graph.

This matters for radius bias specifically. If cost is read off the reference
graph, thinning every vessel by 30% has no functional consequence at all --
which is false: Poiseuille conductance goes as r^4, so it should cost ~76%.
Re-deriving radii from the perturbed mask is what lets a purely geometric
error carry a real functional price, and that is half of the stratified
result the paper is built on.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize

from .skelgraph import skeleton_to_graph, prune_spurs
from . import graphcost as gc


def analyse(mask, prune_px=5, sink_positions=None, source_pos=None, source_positions_all=None):
    """mask -> (graph, source, costs).

    `sink_positions` fixes the perfusion targets to the reference's terminal
    LOCATIONS. Without it, conductance is summed over whatever leaves the
    perturbed graph happens to have, which rewards fragmentation (see
    graphcost.conductance_fixed_sinks).

    `source_pos` pins the source to the reference's source LOCATION (nearest
    graph node). Without it the source is re-picked as the maximum-radius node
    of each perturbed graph, and a break near the reference source moved it to
    another trunk, raising conductance by 26% on STARE im0077 at 20% breaks
    (review panel v1, 2026-09-07). The reference itself is analysed with
    source_pos=None and its source position passed to every perturbed case.
    """
    if mask.sum() == 0:
        return None, None, dict(traceable_length=0.0, conductance=0.0,
                                total_length=0.0, perfused_of_self=0.0)
    skel = skeletonize(mask)
    radius_map = ndimage.distance_transform_edt(mask)
    G, _ = skeleton_to_graph(mask, skel=skel, radius_map=radius_map)
    if G.number_of_edges() == 0:
        return G, None, dict(traceable_length=0.0, conductance=0.0,
                             total_length=0.0, perfused_of_self=0.0)
    G, _ = prune_spurs(G, prune_px)
    if G.number_of_edges() == 0:
        return G, None, dict(traceable_length=0.0, conductance=0.0,
                             total_length=0.0, perfused_of_self=0.0)

    if source_pos is None:
        src = gc.pick_source(G)
    else:
        sp_ = np.asarray(source_pos, dtype=float)
        src = min(G.nodes, key=lambda n: float(np.sum((np.asarray(G.nodes[n]["pos"], dtype=float) - sp_) ** 2)))
    terms = gc.terminals(G)
    total = float(sum(d["length"] for _, _, d in G.edges(data=True)))
    trace = gc.traceable_length(G, src)
    if sink_positions is None:
        cond = gc.conductance(G, src, terms)
        sinks = np.array([G.nodes[t]["pos"] for t in terms], dtype=float) if terms \
            else np.zeros((0, mask.ndim))
    else:
        cond = gc.conductance_fixed_sinks(G, src, sink_positions)
        sinks = np.asarray(sink_positions, dtype=float)
    cond_k = gc.conductance_kirchhoff(G, src, sinks, mask=mask, radius_map=radius_map) if len(sinks) else 0.0
    # multi-root: every reference component's root at unit pressure. Reference call
    # (source_positions_all=None) uses its own component roots and reports them.
    if source_positions_all is None:
        roots = gc.component_sources(G)
    else:
        roots = []
        for rp in np.asarray(source_positions_all, dtype=float).reshape(-1, mask.ndim):
            roots.append(min(G.nodes, key=lambda n: float(np.sum((np.asarray(G.nodes[n]["pos"], dtype=float) - rp) ** 2))))
        roots = list(dict.fromkeys(roots))
    cond_ms = gc.conductance_kirchhoff(G, roots[0], sinks, mask=mask, radius_map=radius_map, extra_sources=roots[1:]) if (len(sinks) and roots) else 0.0
    return G, src, dict(
        traceable_length=trace,
        conductance=cond,
        conductance_k=cond_k,
        conductance_ms=cond_ms,
        source_positions_all=np.array([G.nodes[r]["pos"] for r in roots], dtype=float),
        total_length=total,
        perfused_of_self=trace / total if total > 0 else 0.0,
        sink_positions=sinks,
    )
