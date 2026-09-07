"""Typed, graded error operators -- the generator, exposed as a reusable artifact.

Five operators, each isolating one kind of mistake a segmenter actually makes:

  break     cut branches (connectivity lost, few voxels moved); every cut is
            verified to sever in a local crop, and `info["unsevered"]`
            counts any that could not be
  bridge    join branches that do not touch (connectivity invented)
  truncate  shorten branch endpoints (extent lost)
  radius    thicken or thin uniformly (geometry wrong, topology intact)
  boundary  roughen the surface (voxels moved, nothing structural)

`severity` means "fraction of the eligible population" for break / bridge /
truncate, "fraction of foreground voxels moved" for boundary, and a literal
scale factor for radius (0.7 = 30% thinner, 1.3 = 30% thicker).

The operators are leak-tested: each moves its own kind of error and not the
others. Severities past the tested ladder are not guaranteed leak-free --
radius at scale >= 1.45 dilates vessels into each other and silently changes
Betti-0, which is a topology change wearing a geometry operator's name.
"""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

from ._core import perturb as _p
from ._core import graphcost as _gc
from ._core.skelgraph import skeleton_to_graph, prune_spurs

__all__ = ["perturb", "OPERATORS", "FRAC_LADDER", "RADIUS_LADDER", "BOUNDARY_LADDER"]

OPERATORS = ("break", "bridge", "truncate", "radius", "boundary")

#: Severity ladders used in the paper. Verified leak-free over these ranges.
FRAC_LADDER = _p.FRAC_LADDER
RADIUS_LADDER = (0.5, 0.7, 0.85, 1.15, 1.3)
BOUNDARY_LADDER = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def perturb(mask, operator, severity, *, seed=0, prune_px=5):
    """Apply one graded error operator. Returns (perturbed_mask, info).

    `info` carries what the operator actually did -- how many edges it cut,
    how many voxels it moved -- so a caller can report realised severity
    rather than requested severity. They differ: asking to break 20% of the
    edges of a graph with 7 edges breaks 1.

    Scoring a `radius` perturbation
    -------------------------------
    Radius bias must be scored against its OWN scale-1.0 reconstruction, not
    against the original mask, or the rasterisation error is charged to the
    operator::

        reference, _ = perturb(mask, "radius", 1.0)
        thinned, _   = perturb(mask, "radius", 0.7)
        curvicost.score(reference, thinned)

    The other four operators score against the original mask directly.
    """
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator {operator!r}; expected one of {OPERATORS}")
    mask = np.asarray(mask).astype(bool)
    if mask.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got {mask.ndim}D")
    if not mask.any():
        raise ValueError("mask is empty")

    if operator == "boundary":
        total = int(mask.sum())
        out = _p.boundary_noise_matched(mask, int(severity * total), seed)
        return out, {"voxels_moved": total - int(out.sum()), "requested_frac": severity}

    skel = skeletonize(mask)

    if operator == "radius":
        out, info = _p.radius_bias(mask, severity, skel=skel)
        return out, dict(info, scale=severity)

    graph, _ = skeleton_to_graph(mask, skel=skel)
    graph, _ = prune_spurs(graph, prune_px)
    population = _p.target_population(graph, operator)
    n = _p.n_from_frac(severity, population)

    if operator == "break":
        _, out, info = _gc.break_edges(graph, n, seed, mask=mask)
    elif operator == "bridge":
        out, info = _p.spurious_bridges(mask, graph, n, seed)
    else:  # truncate
        owner = _p.ownership_map(mask, skel)
        out, info = _p.endpoint_truncation(mask, graph, n, seed, skel=skel, owner=owner)

    return out, dict(info, requested_frac=severity, population=population, n_applied=n)
