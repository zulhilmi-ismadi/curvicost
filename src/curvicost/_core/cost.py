"""Functional cost read off the graph, not the raster.

Two costs: one that needs no radii (so it works on neurites and vessels alike)
and one that does (the vessel-native quantity).
"""
from __future__ import annotations

import numpy as np


def _reachable(tree, sources):
    adj = tree.adjacency()
    seen, stack = set(sources), list(sources)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def perfused_fraction(tree, reference_terminals=None, sources=None):
    """Share of the ORIGINAL terminals still reachable from the source.

    Scoring against the reference's terminals is what makes a break expensive:
    an orphaned subtree keeps its own leaves, so grading against the perturbed
    tree's own terminals would hide the damage.
    """
    src = list(tree.roots) if sources is None else list(sources)
    terms = tree.terminals() if reference_terminals is None else list(reference_terminals)
    if not terms:
        return 1.0
    if not src:
        return 0.0
    reach = _reachable(tree, src)
    return float(sum(1 for t in terms if t in reach) / len(terms))


def traceable_length(tree, sources=None):
    """Total edge length still connected to the source (neurite analogue of ERL)."""
    src = list(tree.roots) if sources is None else list(sources)
    if not src:
        return 0.0
    reach = _reachable(tree, src)
    total, e = 0.0, tree.edges
    lens = tree.edge_lengths()
    for k, (c, p) in enumerate(e):
        if int(c) in reach and int(p) in reach:
            total += lens[k]
    return float(total)


def poiseuille_conductance(tree, reference_terminals=None, sources=None, eps=1e-9):
    """Sum over reachable terminals of 1 / sum(L / r^4) along the path.

    Series resistance along each root-to-leaf path, paths in parallel. Terminals
    that no longer connect contribute nothing -- so a mid-trunk break removes
    every path behind it at once.
    """
    src = list(tree.roots) if sources is None else list(sources)
    terms = tree.terminals() if reference_terminals is None else list(reference_terminals)
    if not src or not terms:
        return 0.0

    adj = tree.adjacency()
    parent_of, dist_r = {}, {}
    for s in src:
        parent_of[s], dist_r[s] = None, 0.0
    stack = list(src)
    seen = set(src)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            parent_of[v] = u
            seg = float(np.linalg.norm(tree.xyz[v] - tree.xyz[u]))
            r = max(float(tree.radius[v]), eps)
            dist_r[v] = dist_r[u] + seg / (r ** 4)
            stack.append(v)

    g = 0.0
    for t in terms:
        if t in dist_r and dist_r[t] > 0:
            g += 1.0 / dist_r[t]
    return float(g)


def all_costs(tree, reference):
    """Cost of `tree` graded against the intact `reference` tree.

    Sources and terminals come from the REFERENCE, never from the perturbed
    tree. Breaking an edge makes the orphaned subtree's top node a root, so a
    perturbed tree that supplies its own sources scores its own damage as
    healthy -- the break becomes invisible. (Caught by anchor 1 on first run.)
    Row indices are preserved by Tree.copy(), so reference rows stay valid.
    """
    sources = list(reference.roots)
    terminals = reference.terminals()
    return {
        "perfused_fraction": perfused_fraction(tree, terminals, sources),
        "traceable_length": traceable_length(tree, sources),
        "conductance": poiseuille_conductance(tree, terminals, sources),
    }
