"""The vendored engine must stay byte-identical to the study's code/p16/.

If this fails, the tool and the paper have drifted and any number the tool
prints is no longer the number the paper reports. Re-sync rather than
editing the vendored copy.
"""
import pathlib
import pytest

_HERE = pathlib.Path(__file__).resolve()
VENDORED = _HERE.parents[1] / "src" / "curvicost" / "_core"
STUDY = _HERE.parents[2] / "code" / "p16"

pytestmark = pytest.mark.skipif(
    not STUDY.is_dir(),
    reason="study tree not present (expected when installed from a wheel)",
)


def _modules():
    return sorted(p.name for p in STUDY.glob("*.py"))


def test_no_module_missing():
    vendored = {p.name for p in VENDORED.glob("*.py")}
    assert set(_modules()) <= vendored, f"not vendored: {set(_modules()) - vendored}"


@pytest.mark.parametrize("name", _modules())
def test_module_identical(name):
    if name == "__init__.py":
        pytest.skip("__init__ carries the vendoring notice and differs by design")
    assert (VENDORED / name).read_bytes() == (STUDY / name).read_bytes(), (
        f"{name} has drifted; edit code/p16/{name} then re-run tools/sync_core.sh"
    )
