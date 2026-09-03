"""Synthetic fixtures. No dataset on disk is required to run the test suite."""
import numpy as np
import pytest


def _stamp_line(vol, a, b, radius):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = int(max(abs(b - a).max() * 2, 2))
    grid = np.indices(vol.shape).astype(float)
    for t in np.linspace(0, 1, n):
        centre = a + t * (b - a)
        d2 = sum((grid[i] - centre[i]) ** 2 for i in range(vol.ndim))
        vol |= d2 <= radius ** 2
    return vol


@pytest.fixture(scope="session")
def tree2d():
    """A thick 2D 'Y': one trunk, two branches, three terminals."""
    vol = np.zeros((128, 128), bool)
    _stamp_line(vol, (120, 64), (64, 64), 3.0)   # trunk
    _stamp_line(vol, (64, 64), (16, 30), 2.0)    # branch A
    _stamp_line(vol, (64, 64), (16, 98), 2.0)    # branch B
    return vol


@pytest.fixture(scope="session")
def tree3d():
    """The same topology in 3D, small enough to stay fast."""
    vol = np.zeros((48, 48, 48), bool)
    _stamp_line(vol, (44, 24, 24), (24, 24, 24), 2.5)
    _stamp_line(vol, (24, 24, 24), (6, 10, 24), 2.0)
    _stamp_line(vol, (24, 24, 24), (6, 38, 24), 2.0)
    return vol


@pytest.fixture(scope="session")
def twotrees():
    """Two nearly-touching trees: tips 8 voxels apart, separate components.

    The `bridge` operator needs two degree-1 tips close in SPACE but far in the
    GRAPH; a single tree whose branches diverge cleanly offers no such pair, so
    bridge is legitimately a no-op there. This fixture is the minimal geometry
    that admits every operator.
    """
    vol = np.zeros((160, 160), bool)
    _stamp_line(vol, (150, 50), (100, 50), 3.0)   # tree 1 trunk
    _stamp_line(vol, (100, 50), (60, 30), 2.0)
    _stamp_line(vol, (100, 50), (60, 72), 2.0)    # tip at x=72
    _stamp_line(vol, (10, 110), (60, 110), 3.0)   # tree 2 trunk
    _stamp_line(vol, (60, 110), (60, 80), 2.0)    # tip at x=80 -> 8 voxel gap
    _stamp_line(vol, (10, 110), (20, 140), 2.0)
    return vol
