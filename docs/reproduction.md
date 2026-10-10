# Reproducibility and evaluation protocols

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
hierarchy using CDA-UNet's own predicted parents (`self-parent`). Test labels are
used to calculate metrics after prediction, not to select parent events. A held
out sample must never contribute to model fitting, checkpoint selection, or
construction of the training parents.

This choice permits training from the public dataset without independent UNITO
checkpoints or saved reference predictions. It also makes the training-to-test
change in parent source explicit: training parents are consensus labels,
whereas test parents can contain upstream prediction errors. Report that
difference with any resulting benchmark.

## Relation to the manuscript's reported results

The public 1000-epoch configuration is a standard training configuration, not a
claim that a new run will reproduce the manuscript's main table values. The
following distinctions matter:

| Component | Public default | Existing main-table experiment |
|---|---|---|
| Image grid / adapter width | 101 × 101 / 32 | 101 × 101 / 32 |
| Conditions | Ordered channel identities and gate identity | Same identities |
| Normalization routing | Input-domain encoder BN; per-gate decoder BN | Same routing |
| Training budget | 1000 epochs | 1600 epochs for principal CDA-UNet results |
| Training parents | Consensus parents of training subjects | Saved pooled out-of-fold reference predictions |
| Test parents | Same CDA-UNet model's sequential predictions | Saved independent UNITO reference predictions for the classification and correlation tables |
| Checkpoint distribution | Train your own; no weights supplied | Historical experiment checkpoints are not distributed |

The qualitative manuscript figures used self-parent inference. Their trained
checkpoints and favorable example selection still differ from training the
standard public configuration afresh.

Historical pooled out-of-fold parent construction excluded direct test
subject–gate pairs from optimization. However, a reference model producing a
training subject's parent could itself have been fitted using a subject in the
current outer test fold. This creates an indirect dependency across folds;
the existing numbers are not a strictly nested cross-validation estimate of the
complete gating pipeline. The public workflow does not silently reconstruct or
reuse that cache. To repeat a reference-parent benchmark, record the parent
models, their training membership, and their predictions separately. A strict
nested design requires parent-generating models to use only the current outer
training subjects.

The paper also selected principal configurations from completed experiments by
cross-validation performance. The public defaults are fixed in the released
configuration; they are not claimed to be an independently validated optimum.
No alternative resolutions, ranks, condition ablations, batch-normalization
ablations, or model-selection results are bundled with this release.

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
