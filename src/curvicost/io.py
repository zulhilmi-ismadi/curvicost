"""Mask loading and saving. Deliberately small.

Scope discipline (WP-T): 2D and 3D *binary* curvilinear masks. Anything
non-zero is foreground. Multi-label images, probability maps and time series
are out of scope and are rejected rather than silently thresholded at 0.
"""
from __future__ import annotations

import pathlib

import numpy as np

__all__ = ["load_mask", "save_mask", "voxel_size_of"]

_NIFTI = (".nii", ".nii.gz", ".hdr", ".img")
_TIFF = (".tif", ".tiff")
_PIL = (".png", ".ppm", ".pgm", ".bmp", ".gif", ".jpg", ".jpeg")


def _suffix(path):
    name = pathlib.Path(path).name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return pathlib.Path(name).suffix


def load_mask(path):
    """Load a binary mask. Returns (mask, voxel_size).

    `voxel_size` is a tuple of physical sizes per axis, taken from the NIfTI
    header when present and `None` otherwise. Nothing in the cost model uses
    it yet -- lengths and radii are in voxels -- so it is carried through and
    reported rather than applied. Anisotropic data therefore needs explicit
    handling by the caller; see README "Voxel size".
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ext = _suffix(path)

    if ext in _NIFTI:
        import nibabel as nib
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj)
        vox = tuple(float(z) for z in img.header.get_zooms()[:arr.ndim])
    elif ext in _TIFF:
        import tifffile
        arr = tifffile.imread(str(path))
        vox = None
    elif ext == ".npy":
        arr = np.load(str(path))
        vox = None
    elif ext in _PIL:
        from PIL import Image
        arr = np.array(Image.open(str(path)))
        vox = None
    else:
        raise ValueError(
            f"unsupported mask format {ext!r}. Supported: "
            f"{', '.join(_NIFTI + _TIFF + _PIL + ('.npy',))}"
        )

    # An RGB label image (STARE ships .ppm) collapses to its first channel;
    # a genuine 3D volume must not. Distinguish by trailing-axis size.
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and ext in _PIL:
        arr = arr[..., 0]

    if arr.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if finite.size and not np.isin(finite, (0.0, 1.0)).all():
            raise ValueError(
                f"{path.name} looks like a probability map or a distance field, "
                "not a binary mask. Threshold it first -- curvicost will not "
                "pick a threshold for you, because the choice changes the score."
            )
    n_levels = len(np.unique(arr))
    if n_levels > 2:
        raise ValueError(
            f"{path.name} has {n_levels} distinct values; expected a binary mask. "
            "Multi-label images are out of scope: extract one structure first."
        )
    return arr.astype(bool), vox


def save_mask(mask, path, voxel_size=None):
    """Write a binary mask, format chosen by extension."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = _suffix(path)
    arr = np.asarray(mask).astype(np.uint8)

    if ext in _NIFTI:
        import nibabel as nib
        affine = np.eye(4)
        if voxel_size is not None:
            for i, z in enumerate(voxel_size[:3]):
                affine[i, i] = z
        nib.save(nib.Nifti1Image(arr, affine), str(path))
    elif ext in _TIFF:
        import tifffile
        tifffile.imwrite(str(path), arr * 255)
    elif ext == ".npy":
        np.save(str(path), arr.astype(bool))
    elif ext in _PIL:
        from PIL import Image
        Image.fromarray(arr * 255).save(str(path))
    else:
        raise ValueError(f"unsupported output format {ext!r}")
    return path


def voxel_size_of(spec, ndim):
    """Parse a `--voxel-size` string such as "0.5,0.5,2" into a tuple."""
    if spec is None:
        return None
    parts = [p for p in str(spec).replace("x", ",").split(",") if p.strip()]
    vals = tuple(float(p) for p in parts)
    if len(vals) == 1:
        vals = vals * ndim
    if len(vals) != ndim:
        raise ValueError(f"--voxel-size gave {len(vals)} values for a {ndim}D mask")
    return vals
