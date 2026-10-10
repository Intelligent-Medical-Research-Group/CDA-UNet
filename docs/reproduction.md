# Data preparation and evaluation

This release implements CDA-UNet for the nine-gate mass-cytometry panel and the
seven-gate flow-cytometry panel. The public configurations use 101 × 101 images,
ordered horizontal-channel, vertical-channel and gate identity conditions,
adapter width 32, routed batch normalization, and 1000 training epochs. The
modalities are trained separately. One checkpoint contains every gate of one
modality.

## Dataset and preprocessing

Obtain the published data from
[Pennsieve Discover, dataset 408, version 2](https://discover.pennsieve.io/datasets/408/version/2).
Use the released cell-level CSV measurements and the `GaterConsensus` column for
supervised preparation. The repository contains schema definitions and a loader;
it contains no subject records or annotations. See the main README for the
supported input layout and preparation commands. Retain each complete subject
in a single split; do not split individual events across training and testing.

The released consensus class identifies the deepest attained population.
Preparation expands it to every ancestor in the specified hierarchy. For each
gate, the selected parent events determine the density image and the coordinate
mapping. Both measurement axes are min–max normalized within that parent, then
rounded to integer positions 0 through 100. The first coordinate is reversed to
form the image row. Thus, 101 denotes 101 representable positions per axis. Pixel
counts are divided by the maximum count. Positive consensus events form a filled
convex-hull target in this coordinate system. Targets with at most three occupied
pixels, or a collinear occupied set, retain their occupied pixels. Empty parents
yield no positive child predictions.

Released mass-cytometry marker measurements are already transformed; the loader
does not apply another asinh transform. For flow-image construction, negative
scatter values are clipped to zero, while fluorescence values are unchanged.
Every original event retains its index. A predicted child label is negative for
events excluded by that child's selected parent. Sibling populations use separate
binary predictions and can overlap; this model is not a mutually exclusive
softmax classifier.

## Meaning of routed batch normalization

“Per-gate BN” refers to the normalization routing used in the manuscript's
CDA-UNet architecture. Decoder BN has a separate set of learned affine
parameters and running statistics for each gate. Encoder and bottleneck BN
route by the parent population and ordered channel pair: five domains for mass
cytometry and six for flow cytometry. Sibling gates with identical input
definitions share the encoder domain. This is different from one globally
shared BN throughout the network and from allocating a distinct encoder BN to
every sibling gate. Adapter normalization is affine-free group normalization;
its affine coefficients are supplied by the condition generator.

## Public training and test protocol

The default public workflow uses consensus parent membership to form training
images from training subjects. Test inference proceeds sequentially through the
hierarchy using the events retained by CDA-UNet's preceding gates. Test labels are
used to calculate metrics after prediction, not to select parent events. A held
out sample must never contribute to model fitting, checkpoint selection, or
construction of the training parents.

Both public configurations train for **1000 epochs** and evaluate the final
checkpoint. Their training inputs use consensus populations, while predictions
on a new sample can propagate upstream errors. Record the actual completed
epoch and population selection in every result. Scores from a new run must be
measured with its own checkpoints; the manuscript gives the settings of its
reported experiments separately.

Saved predictions can optionally provide the upstream event selections through
`--parents-dir` or `--parents-file`. Record the generating model and its training
membership. For an independent evaluation, models generating training inputs
must use only subjects from the current outer training fold. File-identity
checks do not establish whether the generating model respected that split.

Only the standard mass and flow configurations are bundled. No trained weights,
alternative model configurations, or benchmark prediction files are included.

## Metric interpretation

Calculate accuracy, recall, precision, and F1 for each subject and gate over all
original events, including events rejected by a parent. Use zero for undefined
precision, recall, or F1. Average subject metrics with equal weight to obtain
each gate's result, and average gate results with equal weight for `Overall`.
Do not calculate mean F1 from mean precision and mean recall, or pool all events
across subjects and call that value the same overall statistic.

Evaluation saves full-precision values in `report.json` and four-decimal summaries
in `per_sample.csv` and `per_gate.csv`. Aggregate the JSON reports so display
rounding does not change the summary. Reports must have disjoint subject and fold
assignments, the same configuration and parent policy, and the same completed
epoch. A subset of folds can be aggregated for progress inspection; it is not a
complete cross-validation result. Check `complete_cross_validation` in the
aggregate report before describing a result as covering every fold.

For population percentages, distinguish a gate's positive-event fraction of all
events from its fraction of its parent. The manuscript's Pearson analyses use
the former; gate-adjacent figure labels use the latter. Correlation with manual
consensus measures association and does not establish interchangeability with
expert gating.

Record the repository commit, configuration, dataset version, split manifest,
random seed, training epoch, parent policy, and software versions alongside any
new result. Hardware and nondeterministic GPU operations can affect numerical
agreement, so exact repeated values should not be assumed.
