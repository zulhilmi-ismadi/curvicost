# curvicost

**What did the segmentation error actually cost?**

Overlap metrics answer *how many voxels did you get right*. For curvilinear
structures — vessels, neurites, airways, cracks — that is usually not the
question anyone is really asking. Two examples from retinal images:

| case | Dice | what the biology loses |
|---|---|---|
| a single spurious bridge between two vessels | **0.9997** | two territories merge |
| two expert annotators, same image | **0.740** | essentially nothing (conductance ratio 1.03) |

Dice ranks these in exactly the wrong order. `curvicost` reports the metric
and the functional cost side by side, so you can see when they agree and
when they do not.

## Install

```bash
pip install curvicost          # core
pip install "curvicost[io]"    # + TIFF / NIfTI / PNG loading
```

## Use

```bash
curvicost score pred.tif --gt gt.tif
```

```
metric                 value   |  cost                    value
------------------------------------------------------------------
dice                  0.9821   |  traceable_frac         0.6042
iou                   0.9648   |  conductance_frac       0.3311
cldice                0.9556   |  perfused_of_self       0.6104
betti0_error               3   |
erl_frac              0.6288   |
diadem_like           0.7415   |
```

From Python:

```python
import curvicost

row = curvicost.score(gt, pred)           # one flat dict: metrics + costs
broken, info = curvicost.perturb(gt, "break", 0.10, seed=0)
```

Generate a graded, typed error from the command line:

```bash
curvicost perturb gt.tif --operator break --severity 0.1 -o broken.tif
```

## What it measures

**Metrics** — Dice, IoU, clDice, Betti-0 error, expected run length (ERL),
and a DIADEM-like matching score.

**Costs** — what an error does to the structure's function, measured on a
graph built from the mask itself:

- `traceable_frac` — fraction of the reference's length still reachable from
  the root. The "can I follow it" cost.
- `conductance_frac` — fraction of Poiseuille conductance retained, summed
  over fixed anatomical sinks. Goes as r⁴, so it is sensitive to thickness in
  a way length is not.
- `perfused_of_self` — how much of the prediction's *own* length it can reach,
  which separates "the mask is fragmented" from "the mask is small".

Cost fractions are relative to the reference's own cost. `1.0` means function
preserved; `0.0` means none of it. **Values above 1.0 are meaningful, not
bugs** — a prediction more liberal than the reference can be more traceable
than it. STARE's two annotators differ by `traceable_frac` 1.95.

## The five error operators

| operator | what it breaks | `severity` means |
|---|---|---|
| `break` | connectivity (few voxels moved) | fraction of edges cut |
| `bridge` | connectivity, invented | fraction of eligible pairs joined |
| `truncate` | extent | fraction of endpoints shortened |
| `radius` | geometry, topology intact | scale factor (0.7 = 30% thinner) |
| `boundary` | surface only | fraction of foreground voxels moved |

Each is leak-tested: it moves its own kind of error and not the others.
Severities beyond the tested ladders are not guaranteed leak-free — `radius`
at scale ≥ 1.45 dilates neighbouring vessels into each other and silently
changes Betti-0, which is a topology change wearing a geometry operator's name.

**Scoring a `radius` perturbation** needs its own scale-1.0 reconstruction as
the reference, or the rasterisation error is charged to the operator:

```python
reference, _ = curvicost.perturb(mask, "radius", 1.0)
thinned, _   = curvicost.perturb(mask, "radius", 0.7)
curvicost.score(reference, thinned)
```

The other four score against the original mask directly.

## Two invariants

These are load-bearing. They are the difference between a cost model and a
number that looks like one.

1. **The reference and the prediction take the same path.** Both are
   skeletonised and re-graphed from their own mask; nothing is inherited from
   the reference graph. Read cost off the reference graph instead and thinning
   every vessel by 30% costs nothing — which is false, since conductance goes
   as r⁴.
2. **Sinks are fixed to the reference's terminal locations.** Otherwise
   conductance is summed over whatever leaves the *predicted* graph happens to
   have, and shattering a tree manufactures new low-resistance leaves near the
   root — fragmentation scores *better*.

## Scope

2D and 3D **binary** curvilinear masks. Nothing else. Probability maps and
multi-label images are rejected rather than silently thresholded, because the
threshold choice changes the score and it is not this tool's to make.

### Voxel size

Lengths and radii are reported in **voxels**, not physical units. A voxel size
is read from NIfTI headers and carried into the output for the record, but it
is not yet applied to the cost model. Anisotropic data therefore needs
resampling to isotropic before scoring, or the r⁴ conductance term is wrong by
the anisotropy ratio to the fourth power. `curvicost score` warns when it sees
anisotropic voxels.

## Tests

```bash
pip install "curvicost[test]" && pytest
```

## Licence

MIT.
