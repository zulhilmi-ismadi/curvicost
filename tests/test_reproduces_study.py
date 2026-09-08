"""The tool must reproduce the study's own recorded numbers.

`test_core_sync.py` proves the engine source has not drifted. This proves the
package *composes* that engine the way the study scripts do -- same reference
handling, same fixed sinks, same pinned source, same reference-skeleton reach,
same ERL reference -- by re-scoring cases whose answers are already on disk in
`results/realistic_errors_stare_v7.parquet` (the study's data of record: v6 costs
with the root fallback and the two-sided conductance; earlier tables are superseded).

Skipped when the study tree is absent (i.e. installed from a wheel).
"""
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()
STUDY = _HERE.parents[2]
PARQUET = STUDY / "results" / "realistic_errors_stare_v7.parquet"
LABELS = STUDY / "data" / "stare"

pytestmark = pytest.mark.skipif(
    not (PARQUET.exists() and LABELS.is_dir()),
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
    "traceable_ms_frac": "traceable_frac",          # cost of record: multi-root reach
    "traceable_frac": "traceable_single_frac",      # single pinned source
    "conductance_frac": "conductance_frac",         # Kirchhoff, fixed reference sinks
    "conductance_twosided": "conductance_twosided", # the reported flow cost, min(c, 1/c)
}


@pytest.mark.parametrize("image", ["im0001", "im0077"])
def test_matches_recorded_observer_row(image):
    pd = pytest.importorskip("pandas")
    import curvicost

    ah = LABELS / "labels-ah" / f"{image}.ah.ppm"
    vk = LABELS / "labels-vk" / f"{image}.vk.ppm"
    if not (ah.exists() and vk.exists()):
        pytest.skip(f"{image} annotations not present")

    recorded = pd.read_parquet(PARQUET)
    row = recorded[(recorded.image == image) & (recorded.source == "observer")]
    assert len(row) == 1, f"expected exactly one recorded observer row for {image}"
    row = row.iloc[0]

    gt, _ = curvicost.load_mask(ah)
    pred, _ = curvicost.load_mask(vk)
    got = curvicost.score(gt, pred)

    for column, key in COLUMNS.items():
        assert got[key] == pytest.approx(float(row[column]), rel=1e-6), (
            f"{key}: tool {got[key]!r} != recorded {row[column]!r}. The package no "
            f"longer reproduces the study pipeline."
        )


def test_prepared_reference_gives_identical_numbers():
    """`prepare_reference` is an optimisation, not a second code path."""
    import curvicost

    ah = LABELS / "labels-ah" / "im0001.ah.ppm"
    vk = LABELS / "labels-vk" / "im0001.vk.ppm"
    if not (ah.exists() and vk.exists()):
        pytest.skip("im0001 annotations not present")
    gt, _ = curvicost.load_mask(ah)
    pred, _ = curvicost.load_mask(vk)
    direct = curvicost.score(gt, pred)
    ctx = curvicost.prepare_reference(gt)
    via_ctx = curvicost.score(gt, pred, reference=ctx)
    for key in curvicost.COST_COLUMNS + ("dice", "erl_frac", "diadem_like"):
        assert via_ctx[key] == pytest.approx(direct[key], rel=0, abs=0), key
