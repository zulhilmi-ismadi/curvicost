"""SWC trace -> graph -> voxel mask.

The graph is the primary representation: perturbations act on it, costs are read
from it, and masks are rasterised from it. Keeping the ground truth and the
perturbed case on the same path means no skeletonisation enters the loop, so a
failed sanity check points at this code rather than at a skeletoniser.
"""
from __future__ import annotations

import numpy as np

SWC_COLS = ("id", "type", "x", "y", "z", "radius", "parent")


class Tree:
    """A neurite trace: nodes with coordinates and radii, edges from parent links."""

    def __init__(self, ids, xyz, radius, parent):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.radius = np.asarray(radius, dtype=np.float64)
        self.parent = np.asarray(parent, dtype=np.int64)
        self._index = {int(i): k for k, i in enumerate(self.ids)}

    # --- construction -----------------------------------------------------
    @classmethod
    def from_swc(cls, path):
        rows = np.loadtxt(path, comments="#")
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.shape[1] < 7:
            raise ValueError(f"{path}: expected >=7 SWC columns, got {rows.shape[1]}")
        return cls(rows[:, 0], rows[:, 2:5], rows[:, 5], rows[:, 6])

    def copy(self):
        return Tree(self.ids.copy(), self.xyz.copy(), self.radius.copy(), self.parent.copy())

    # --- topology ---------------------------------------------------------
    @property
    def edges(self):
        """(child_row, parent_row) pairs. Root links (parent -1) are not edges."""
        out = []
        for k, p in enumerate(self.parent):
            if p == -1:
                continue
            j = self._index.get(int(p))
            if j is not None:
                out.append((k, j))
        return np.asarray(out, dtype=np.int64).reshape(-1, 2)

    @property
    def roots(self):
        return np.flatnonzero(self.parent == -1)

    def adjacency(self):
        adj = {k: [] for k in range(len(self.ids))}
        for c, p in self.edges:
            adj[int(c)].append(int(p))
            adj[int(p)].append(int(c))
        return adj

    def edge_lengths(self):
        e = self.edges
        if len(e) == 0:
            return np.zeros(0)
        return np.linalg.norm(self.xyz[e[:, 0]] - self.xyz[e[:, 1]], axis=1)

    def terminals(self):
        """Leaf nodes: degree 1 and not a root. Roots of size-1 components excluded."""
        adj = self.adjacency()
        return [k for k, nb in adj.items() if len(nb) == 1 and self.parent[k] != -1]

    def components(self):
        """Connected components as lists of row indices (undirected)."""
        adj = self.adjacency()
        seen, comps = set(), []
        for start in range(len(self.ids)):
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        return comps

    def total_length(self):
        return float(self.edge_lengths().sum())


def rasterise(tree, shape=None, origin=None, scale=1.0, radius_scale=1.0, min_radius=0.6):
    """Draw the tree into a binary volume by stamping spheres along each edge.

    Sampling step is half a voxel so consecutive stamps overlap; this makes the
    mask's connectivity follow the graph's rather than the sampling rate's.
    """
    xyz = tree.xyz * scale
    rad = np.maximum(tree.radius * scale * radius_scale, min_radius)

    if origin is None:
        origin = xyz.min(axis=0) - (rad.max() + 2)
    xyz = xyz - origin
    if shape is None:
        shape = tuple(int(np.ceil(v)) + int(rad.max() + 3) for v in xyz.max(axis=0))

    vol = np.zeros(shape, dtype=bool)
    edges = tree.edges
    segs = ([(xyz[c], xyz[p], rad[c], rad[p]) for c, p in edges] if len(edges)
            else [(xyz[k], xyz[k], rad[k], rad[k]) for k in range(len(xyz))])
    # isolated nodes still deserve a blob, so they are not silently dropped
    if len(edges):
        linked = set(edges.ravel().tolist())
        segs += [(xyz[k], xyz[k], rad[k], rad[k])
                 for k in range(len(xyz)) if k not in linked]

    for a, b, ra, rb in segs:
        n = max(int(np.ceil(np.linalg.norm(b - a) * 2)), 1)
        for t in np.linspace(0.0, 1.0, n + 1):
            _stamp(vol, a + (b - a) * t, ra + (rb - ra) * t, shape)
    return vol, origin


def _stamp(vol, centre, r, shape):
    lo = np.maximum(np.floor(centre - r).astype(int), 0)
    hi = np.minimum(np.ceil(centre + r).astype(int) + 1, shape)
    if np.any(lo >= hi):
        return
    grids = np.ogrid[tuple(slice(l, h) for l, h in zip(lo, hi))]
    d2 = sum((g - c) ** 2 for g, c in zip(grids, centre))
    vol[tuple(slice(l, h) for l, h in zip(lo, hi))] |= d2 <= r * r
