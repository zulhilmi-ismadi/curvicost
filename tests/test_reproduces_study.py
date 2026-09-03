"""The tool must reproduce the study's own recorded numbers.

`test_core_sync.py` proves the engine source has not drifted. This proves the
package *composes* that engine the way the study scripts do -- same reference
handling, same fixed sinks, same ERL reference -- by re-scoring a case whose
answer is already on disk in `results/realistic_errors_stare.parquet`.

Skipped when the study tree is absent (i.e. installed from a wheel).
"""
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()
STUDY = _HERE.parents[2]
PARQUET = STUDY / "results" / "realistic_errors_stare.parquet"
AH = STUDY / "data" / "stare" / "labels-ah" / "im0001.ah.ppm"
VK = STUDY / "data" / "stare" / "labels-vk" / "im0001.vk.ppm"

pytestmark = pytest.mark.skipif(
    not (PARQUET.exists() and AH.exists() and VK.exists()),
    reason="study data/results not present",
)

# Column name in the parquet -> key returned by curvicost.score()
COLUMNS = {
    "m_dice": "dice",
    "m_iou": "iou",
    "m_cldice": "cldice",
    "m_betti0_error": "betti0_error",
    "m_erl_frac": "erl_frac",
    "m_diadem_like": "diadem_like",
    "traceable_frac": "traceable_frac",
    "conductance_frac": "conductance_frac",
}


def test_matches_recorded_observer_row():
    pd = pytest.importorskip("pandas")
    import curvicost

    recorded = pd.read_parquet(PARQUET)
    row = recorded[(recorded.image == "im0001") & (recorded.source == "observer")]
    assert len(row) == 1, "expected exactly one recorded observer row for im0001"
    row = row.iloc[0]

    gt, _ = curvicost.load_mask(AH)
    pred, _ = curvicost.load_mask(VK)
    got = curvicost.score(gt, pred)

    for column, key in COLUMNS.items():
        assert got[key] == pytest.approx(float(row[column]), rel=1e-6), (
            f"{key}: tool {got[key]!r} != recorded {row[column]!r}. The package no "
            f"longer reproduces the study pipeline."
        )
