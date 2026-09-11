# curvicost

**What did the segmentation error actually cost?**

Overlap metrics answer *how many voxels did you get right*. For curvilinear
structures — vessels, neurites, airways, cracks — that is usually not the
question anyone is really asking. Two examples from one retinal image:

| case | Dice | what the biology loses |
|---|---|---|
| the vessel tree cut at 10% of its branches (under 1% of pixels moved) | **0.993** | about half of the reference skeleton is no longer reachable (0.54) |
| two expert annotators, same image | **0.741** | almost nothing: 98% of the reference skeleton stays reachable |

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
curvicost perturb gt.png --operator break --severity 0.10 --prune-px 17 -o broken.png
curvicost score broken.png --gt gt.png --prune-px 17 --json row.json
```

```
metric                 value   |  cost                      value
--------------------------------------------------------------------
dice                 0.9897   |  traceable_frac           0.8484
iou                  0.9795   |  traceable_single_frac    0.8429
cldice               0.9903   |  conductance_twosided     0.1949
betti0_error              9   |  conductance_frac         0.1949
erl_frac             0.7287   |  perfused_of_self         0.8304
diadem_like          0.9577   |
```

(Output of curvicost 0.2.0 on FIVES test mask `100_D` with 10% of its skeleton
branches cut, 17 breaks: 0.14% of the pixels moved, Dice 0.99, 15% of the
reference skeleton no longer reachable and 80% of its hydraulic conductance
gone. `--prune-px 17` is the scale for these 2048-pixel fundus masks; see
"Resolution" below.)

From Python:

```python
import curvicost

row = curvicost.score(gt, pred)           # one flat dict: metrics + costs
broken, info = curvicost.perturb(gt, "break", 0.10, seed=0)

ctx = curvicost.prepare_reference(gt)     # scoring many predictions against one reference
rows = [curvicost.score(gt, p, reference=ctx) for p in predictions]
```

Generate a graded, typed error from the command line:

```bash
curvicost perturb gt.tif --operator break --severity 0.1 -o broken.tif
```

## Which error types is your metric blind to?

Whether a metric predicts functional cost depends on the error type, the
structure class **and** the cost — so a published table for retina does not
transfer to neurites. `audit` recomputes the map on your own reference masks:

```bash
curvicost audit gt1.png gt2.png gt3.png gt4.png --prune-px 17 --csv blindness.csv
```

```
81 cases from 4 reference mask(s)

  vs traceable length (reachable reference skeleton)
  metric              break     bridge   truncate     radius   boundary
  dice             (+0.85)?      undef   (+0.96)?   (+0.47)?   (+0.78)?
  iou              (+0.85)?      undef   (+0.96)?   (+0.47)?   (+0.78)?
  cldice           (+0.80)?      undef   (+1.00)?   (+0.37)?   (+1.00)?
  betti0_error     (+0.87)?      undef      undef      undef      undef
  erl_frac         (+0.98)?      undef   (+1.00)?   (+1.00)?   (+1.00)?
  diadem_like      (+0.65)?      undef   (+0.92)?   (-0.71)?   (+0.91)?

  every cell: sign-aligned Spearman ρ, 95% cluster-bootstrap interval, acting cases / acting units, label

  vs traceable length (reachable reference skeleton)
  operator  metric              ρ  [   lo,    hi]  cases  units  label
  break     dice           +0.850  [ +0.77,  +0.96]     20      4  inconclusive (<10 units)
  break     cldice         +0.802  [ +0.70,  +0.96]     20      4  inconclusive (<10 units)
  break     erl_frac       +0.979  [ +0.92,  +1.00]     20      4  inconclusive (<10 units)
  bridge    dice            undef  [    --,    --]      5      1  undefined
  radius    cldice         +0.366  [ -0.09,  +0.81]     16      4  inconclusive (<10 units)
  radius    diadem_like    -0.712  [ -0.76,  -0.70]     16      4  inconclusive (<10 units)
  ...

  scale: median vessel radius 5.9 px; --prune-px 17 = 3.0 vessel radii.

  labels (60 cells; definitions as in the paper's Methods):
  undef  undefined — the cost or the metric is constant across the cell, so ρ does not exist   (18)
  (ρ)?   inconclusive — the operator acted in fewer than 10 units or fewer than 12 cases; ρ is shown
         for the record but is NOT a validated correlation and is never read as blind or anti-correlated   (42)
  *      blind — the interval includes zero: the metric cannot see that error type on this data   (0)
  !      anti-correlated — ρ is negative and the interval excludes zero: the metric moves the WRONG WAY   (0)
         tracks — ρ is positive and the interval excludes zero   (0)
```

(curvicost 0.2.0 on four full-resolution FIVES masks, CC BY 4.0; the
conductance block, most of the per-cell table and the per-operator notes are
omitted here.)

Read it as a warning list. Every (cost, operator, metric) cell carries one of
four labels, defined exactly as in the paper's Methods, and the same label is
written to the JSON and the CSV:

| label | printed as | definition |
|---|---|---|
| **undefined** | `undef` | the cost or the metric is constant across the cell, so no correlation exists. A statement about the cost (or the metric), not about blindness: a cost that an error type cannot change is the wrong cost for that error type, and no choice of metric fixes that. |
| **inconclusive** | `(ρ)?` | the operator acted in fewer than **10 units** (reference masks) or fewer than **12 cases**. ρ is printed in parentheses for the record but is *not* a validated correlation and is never read as blind or anti-correlated. |
| **blind** | `*` | the 95% cluster-bootstrap interval includes zero: that metric cannot see that error type on your data. Report it and you are reporting nothing about that failure mode. |
| **anti-correlated** | `!` | the sign-aligned ρ is negative *and* the interval excludes zero: the metric moves the wrong way. Check first whether the *cost* is the right one for that error type. |

A cell that is none of these *tracks* the cost. Correlations are sign-aligned,
so more positive always means "tracks the cost", whichever direction the raw
metric runs. After the grid the report lists every cell in full — ρ, its
interval, acting cases and acting units — and the notes under it say why a
column is undefined (on these thick vessels the bridge operator finds one
eligible tip pair in one image, so every severity yields the same mask).

**Units are reference masks, and the floors are the paper's.** The 10-unit and
12-case floors are counted per cell, over the masks and cases in which the
operator actually changed the mask, so a verdict of any kind needs at least
ten masks; the FIVES example above has four, which is why every defined cell
there is inconclusive. Below three masks the tool reports correlations without
intervals and says so rather than inventing them. Each operator also needs at
least three cases; the report names any it had to skip, and any whose severity
ladder saturated on a small eligible population.

**The default ladder is shortened.** An audit is a diagnostic run on your own
machine, so by default break/bridge/truncate run at 2, 5, 10, 20 and 50 % of
the eligible population (5 rungs), radius at ×0.7, 0.85, 1.15 and 1.3
(4 factors), boundary at 0.5, 2, 5, 10 and 20 % of foreground voxels
(5 levels), with one seed and a 1,000-iteration cluster bootstrap. The study
used 7 rungs at 1, 2, 5, 10, 20, 35 and 50 %, radius ×0.5/0.7/0.85/1.15/1.3,
boundary 0.2/0.5/1/2/5/10/20 %, 3 seeds and 2,000 iterations.
`--severities` widens break/bridge/truncate **only**; `--radius-scales`,
`--boundary-fracs`, `--seeds` and `--bootstrap` set the rest, and
`--study-ladder` sets all of them to the study's values at once:

```bash
curvicost audit gt*.png --prune-px 17 --study-ladder --csv blindness.csv
```

## Which error types does *your* method make?

The audit tells you which metric to distrust for each error type. Acting on it
needs one more thing the audit cannot supply: knowing which error types your own
method produces, because a real prediction is a mixture and arrives unlabelled.
`profile` estimates the mixture — it expresses the disagreement as non-negative
shares of the five operators, calibrated on **your** reference at the severity
whose total disagreement matches:

```bash
curvicost profile pred1.png pred2.png --gt reference.png --json profile.json
```

```
1 prediction(s) of ONE method against one reference

  prediction                     break    bridge  truncate    radius  boundary  unexplained
  annotator2.npy                    6%       32%        0%       37%       25%        37% +

  Read the 'radius' column of `curvicost audit` first: it is the error type these
  predictions most resemble on this data. Shares are coarse -- read them as
  "mostly radius-like", not as a measurement.

  + more than 30% of the disagreement is explained by no combination of the five
    operators. The commonest cause is over-tracing: only `bridge` adds structure,
    so a method that paints more than the reference falls outside the model and the
    audit's columns describe it only in part.
```

(STARE im0077, the second expert annotator scored against the first.)

Pass predictions from **one** method; the mean row is meaningless across
different methods. Each pair is reduced to a signature of seven quantities the
operators move differently — voxels removed, voxels added, change in component
count, merged fraction, reachable-length loss, median-radius ratio and terminal
loss — and the observed signature is fitted as a non-negative combination of the
five operator signatures. Each basis signature averages three realisations of
its operator (`--seeds`), because one draw is not the operator: at a fixed break
severity, which edges the draw happens to cut moves the reachable-length loss by
a factor of five.

**Read `unexplained` first.** It is the part of the disagreement that no
combination of the five operators reproduces. Handed a prediction that *is* one
operator, the profiler names it in 86 of 93 cases across 20 fundus images; on
real methods the residual is much larger, and a large residual means the model
does not span what your method does. The shares are still the right place to
start reading the audit, but they are not the whole error.


## What it measures

**Metrics** — Dice, IoU, clDice, Betti-0 error, expected run length (ERL),
and a DIADEM-like matching score.

**Costs** — what an error does to the structure's function. The reference is
skeletonised once; the prediction is skeletonised and re-graphed from its own
mask.

- `traceable_frac` — **reachable length.** Every connected component of the
  reference is traced from its own root (the skeleton voxel of maximum
  distance-transform value); the cost counts the reference-skeleton voxels
  that lie in the prediction's connected component containing that root. The
  "can I follow it" cost. It is bounded by 1: a prediction cannot be more
  reachable than the reference, and a false connection between two reference
  fragments adds nothing (their voxels were already counted).
- `traceable_single_frac` — the same count from one pinned source, the
  reference's maximum-radius node. On a fragmented reference this can exceed 1
  (a bridge lets one source reach another fragment), which is why the
  multi-root variant is the default.
- `conductance_frac` — **hydraulic conductance retained**, from a Kirchhoff
  (resistor-network) solve on the prediction's skeleton graph: unit pressure at
  the reference's source, zero pressure at every reference terminal that
  survives in the prediction, resistance integrated pixel by pixel as 1/r⁴
  from the distance transform. Goes as r⁴, so it is sensitive to thickness in a
  way length is not. Adding a path can never lower it (Rayleigh monotonicity),
  so a prediction thicker than the reference can exceed 1.
- `perfused_of_self` — how much of the prediction's *own* skeleton length it
  can reach from the pinned source, which separates "the mask is fragmented"
  from "the mask is small".

Cost fractions are relative to the reference's own cost. `1.0` means function
preserved; `0.0` means none of it.

## The five error operators

| operator | what it breaks | `severity` means |
|---|---|---|
| `break` | connectivity (few voxels moved); every cut verified to sever | fraction of edges cut |
| `bridge` | connectivity, invented | fraction of eligible pairs joined |
| `truncate` | extent | fraction of endpoints shortened |
| `radius` | geometry, topology intact | scale factor (0.7 = 30% thinner) |
| `boundary` | surface only | fraction of foreground voxels moved |

Each is leak-tested: it moves its own kind of error and not the others.
Severities beyond the tested ladders are not guaranteed leak-free — `radius`
at scale ≥ 1.45 dilates neighbouring vessels into each other and silently
changes Betti-0, which is a topology change wearing a geometry operator's name.

`break` sizes each gap from the mask's own distance transform and grows it
until a local crop confirms the branch is severed; `info["unsevered"]` counts
any cut that could not be, so a leak is reported rather than silent.

**Scoring a `radius` perturbation** needs its own scale-1.0 reconstruction as
the reference, or the rasterisation error is charged to the operator:

```python
reference, _ = curvicost.perturb(mask, "radius", 1.0)
thinned, _   = curvicost.perturb(mask, "radius", 0.7)
curvicost.score(reference, thinned)
```

The other four score against the original mask directly.

## Four invariants

These are load-bearing. They are the difference between a cost model and a
number that looks like one.

1. **The reference and the prediction take the same path.** Both are
   skeletonised and re-graphed from their own mask; nothing is inherited from
   the reference graph. Read cost off the reference graph instead and thinning
   every vessel by 30% costs nothing — which is false, since conductance goes
   as r⁴.
2. **Sinks are fixed to the reference's terminal locations**, and a sink
   survives only if it is still connected to the source through the
   prediction's mask. Otherwise conductance is summed over whatever leaves the
   *predicted* graph happens to have, and shattering a tree manufactures new
   low-resistance leaves near the root — fragmentation scores *better*.
3. **The source is pinned to the reference's source location.** Re-picking the
   maximum-radius node on every prediction let a break near the source move it
   to another trunk and *raise* conductance.
4. **Reachable length is counted on the reference skeleton**, through the
   prediction's connectivity, never on a re-skeletonised prediction. The cut
   faces of a break re-skeletonise into extra length, which is how the previous
   graph-based definition read above 1 on half the vascular break cases.

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

### Resolution

`--prune-px` (spur pruning, default 5) and the bridge operator's 12-pixel gap
are absolute pixel lengths. `audit` reports the pruning length in units of
your structure's own median vessel radius and warns outside 1–4 radii; the
study's four datasets sit at 2.0–2.7. Scale it to your data (the FIVES example
above uses 17 px on 5.9 px vessels) or the numbers are not comparable.

## What changed in 0.2

The cost definitions were replaced after a review-panel diagnostic on the
accompanying study found four defects in the 0.1 costs (a break operator that
sometimes only constricted, a source that moved under perturbation, a
traceable length inflated by re-skeletonised cut faces, and a path-sum
conductance that charged a bridge as a flow loss). `traceable_frac` is now the
multi-root reachable length and `conductance_frac` the Kirchhoff conductance
described above; `traceable_single_frac` is new. Numbers from 0.1 are not
comparable with numbers from 0.2 and must not be mixed in one table.

`audit` gained `--cost`, and its statistics now follow the paper's Methods
exactly: every cell is labelled *undefined*, *inconclusive*, *blind* or
*anti-correlated* as defined above, the 10-unit and 12-case floors are counted
per cell over the units and cases in which the operator acted, and a cell below
a floor is inconclusive whatever its interval says (0.2.0 counted the unit floor
once over the whole audit and applied it only to intervals spanning zero, and
called a cell anti-correlated only below ρ = −0.2; both are gone). The report
prints every cell with its interval, acting cases and acting units, and the
study's full ladder is reachable from the command line (`--study-ladder`).
`profile` is new: it names which of the five error types a method's own
predictions resemble, which is what makes "stratify by error type" something a
user can act on rather than advice. The
vendored engine is byte-identical to the study's code, and
`tests/test_reproduces_study.py` re-scores study cases against the archived
tables to 10⁻⁶.

## Tests

```bash
pip install "curvicost[test]" && pytest
```

## Releasing

Releases are published by GitHub Actions through PyPI's Trusted Publishing, so no
API token is stored anywhere. One-time setup on pypi.org → project `curvicost` →
Publishing → add a GitHub publisher with owner `zulhilmi-ismadi`, repository
`curvicost`, workflow `publish.yml`, environment `pypi`; and on GitHub create the
`pypi` environment under Settings → Environments.

To release: bump `version` in `pyproject.toml`, commit, then

```bash
git tag v0.2.0 && git push origin v0.2.0
```

The workflow refuses to publish if the tag and the package version disagree.

## Licence

MIT.
