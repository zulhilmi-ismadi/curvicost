"""curvicost -- what did the segmentation error actually cost?

Overlap metrics answer "how many voxels did you get right". For curvilinear
structures -- vessels, neurites, airways, cracks -- that is not the question
anyone is really asking. A bridge between two vessels moves 0.06% of the
voxels and reads Dice 0.9997 while merging two territories; two expert
annotators disagree at Dice 0.740 and lose nothing functional at all.

curvicost puts the metric and the cost side by side:

    >>> import curvicost
    >>> row = curvicost.score(gt, pred)
    >>> row["dice"], row["conductance_frac"]

and generates graded, typed errors to calibrate against:

    >>> broken, info = curvicost.perturb(gt, "break", 0.10, seed=0)

Scope: 2D and 3D binary curvilinear masks. Nothing else.
"""
from __future__ import annotations

__version__ = "0.1.0.dev0"

from .score import score, METRIC_COLUMNS, COST_COLUMNS
from .perturb import perturb, OPERATORS
from .audit import audit, sweep
from .io import load_mask, save_mask

__all__ = [
    "score", "perturb", "audit", "sweep", "load_mask", "save_mask",
    "METRIC_COLUMNS", "COST_COLUMNS", "OPERATORS", "__version__",
]
