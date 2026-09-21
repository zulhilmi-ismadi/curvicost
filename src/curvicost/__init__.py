"""curvicost -- what did the segmentation error actually cost?

Overlap metrics answer "how many voxels did you get right". For curvilinear
structures -- vessels, neurites, airways, cracks -- that is not the question
anyone is really asking. Cutting a retinal vessel tree at 10% of its branches
moves under 1% of the pixels and reads Dice 0.99 while roughly half of the
reference skeleton stops being reachable; two expert annotators disagree at
Dice 0.74 and keep 98% of it reachable.

curvicost puts the metric and the cost side by side:

    >>> import curvicost
    >>> row = curvicost.score(gt, pred)
    >>> row["dice"], row["traceable_frac"], row["conductance_frac"]

and generates graded, typed errors to calibrate against:

    >>> broken, info = curvicost.perturb(gt, "break", 0.10, seed=0)

Scope: 2D and 3D binary curvilinear masks. Nothing else.
"""
from __future__ import annotations

__version__ = "0.2.2"

from .score import score, prepare_reference, METRIC_COLUMNS, COST_COLUMNS
from .perturb import perturb, OPERATORS
from .audit import audit, sweep
from .profile import profile
from .io import load_mask, save_mask

__all__ = [
    "score", "prepare_reference", "perturb", "audit", "sweep", "profile", "load_mask", "save_mask",
    "METRIC_COLUMNS", "COST_COLUMNS", "OPERATORS", "__version__",
]
