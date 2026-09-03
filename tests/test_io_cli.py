import json

import numpy as np
import pytest

from curvicost.cli import main
from curvicost.io import load_mask, save_mask, voxel_size_of


@pytest.mark.parametrize("ext", [".npy", ".tif", ".png"])
def test_roundtrip(tmp_path, tree2d, ext):
    path = save_mask(tree2d, tmp_path / f"m{ext}")
    back, _ = load_mask(path)
    assert np.array_equal(back, tree2d)


def test_nifti_roundtrip_keeps_voxel_size(tmp_path, tree3d):
    nib = pytest.importorskip("nibabel")
    path = save_mask(tree3d, tmp_path / "m.nii.gz", voxel_size=(0.5, 0.5, 2.0))
    back, vox = load_mask(path)
    assert np.array_equal(back, tree3d)
    assert vox == pytest.approx((0.5, 0.5, 2.0))


def test_probability_map_rejected(tmp_path):
    path = tmp_path / "p.npy"
    np.save(path, np.random.default_rng(0).random((32, 32)))
    with pytest.raises(ValueError, match="probability map"):
        load_mask(path)


def test_multilabel_rejected(tmp_path):
    path = tmp_path / "l.npy"
    np.save(path, np.arange(9).reshape(3, 3).astype(np.uint8))
    with pytest.raises(ValueError, match="binary"):
        load_mask(path)


def test_unsupported_format_rejected(tmp_path):
    path = tmp_path / "x.xyz"
    path.write_bytes(b"nope")
    with pytest.raises(ValueError, match="unsupported"):
        load_mask(path)


def test_voxel_size_parsing():
    assert voxel_size_of("0.5,0.5,2", 3) == (0.5, 0.5, 2.0)
    assert voxel_size_of("0.5", 3) == (0.5, 0.5, 0.5)
    assert voxel_size_of(None, 3) is None
    with pytest.raises(ValueError):
        voxel_size_of("1,2", 3)


def test_cli_score_writes_json(tmp_path, tree2d, capsys):
    gt = save_mask(tree2d, tmp_path / "gt.npy")
    out = tmp_path / "row.json"
    rc = main(["score", str(gt), "--gt", str(gt), "--json", str(out), "--quiet"])
    assert rc == 0
    row = json.loads(out.read_text())
    assert row["dice"] == pytest.approx(1.0)
    assert row["conductance_frac"] == pytest.approx(1.0, abs=1e-9)


def test_cli_score_prints_json_by_default(tmp_path, tree2d, capsys):
    gt = save_mask(tree2d, tmp_path / "gt.npy")
    assert main(["score", str(gt), "--gt", str(gt), "--no-erl"]) == 0
    assert json.loads(capsys.readouterr().out)["dice"] == pytest.approx(1.0)


def test_cli_perturb_then_score(tmp_path, tree2d):
    gt = save_mask(tree2d, tmp_path / "gt.npy")
    broken = tmp_path / "broken.npy"
    assert main(["perturb", str(gt), "--operator", "break", "--severity", "0.5",
                 "-o", str(broken), "--quiet"]) == 0
    out = tmp_path / "row.json"
    assert main(["score", str(broken), "--gt", str(gt), "--json", str(out),
                 "--quiet"]) == 0
    assert json.loads(out.read_text())["traceable_frac"] < 1.0


def test_cli_missing_file_is_clean_error(tmp_path, capsys):
    rc = main(["score", str(tmp_path / "nope.npy"), "--gt", str(tmp_path / "nope.npy")])
    assert rc == 2
    assert "curvicost:" in capsys.readouterr().err
