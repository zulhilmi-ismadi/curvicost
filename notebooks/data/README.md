# Example masks for the tutorial

Four vessel masks from the **FIVES** retinal dataset test split, downsampled 3x by
mean-pooling (threshold 0.5) from 2048x2048 to 682x682 so the tutorial runs in minutes.

FIVES is released under **CC BY 4.0**. Please cite:

> Jin, K. *et al.* FIVES: a fundus image dataset for artificial intelligence based vessel
> segmentation. *Scientific Data* **9**, 475 (2022). https://doi.org/10.1038/s41597-022-01564-3

The downsampling is deliberate and is discussed in the tutorial: at full resolution FIVES
vessels are ~6.7 px wide and curvicost's default spur pruning (5 px) is only 0.75 vessel
radii, which the tool warns about. At 3x downsampling they are ~2.2 px, comparable to the
datasets the method was calibrated on.
