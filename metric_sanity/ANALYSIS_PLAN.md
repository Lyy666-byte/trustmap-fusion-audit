# Analysis plan: causal test of IVIF metrics (written 2026-10-31, before any result below was computed)

Already seen when this plan was written: the observational EVAFusion pilot (results/eva_pilot.csv, eva_checks.csv).
Not yet seen: every result of the experiments defined here. This file is local, not a public registration, and
must not be called a preregistration unless it is deposited on a public registry before the human data are collected.

## Question
When the same fused image is changed by one known transformation, do the objective metrics, a downstream
segmentation task and human preference move together?

## Data
- Methods with full MSRS test outputs: DCEvo, EMMA, LUT-Fuse, MMDRFuse.
- Segmentation: all 361 MSRS test pairs (labels in tmp_msrs/test/Segmentation_labels).
- Metrics: 100 MSRS images drawn with seed 20261031 and all conditions, plus the 30 RoadScene images used for humans.
- Humans: 60 of those 100 MSRS images (same seed) and 30 RoadScene images (same seed).
- Images are never chosen by looking at metric, task or human outcomes.

## Conditions (on the luminance Y in [0,1], then clipped and saved as 8 bit)
Native F; blend a=.25/.5/.75, F_a=(1-a)F+a(IR+VIS)/2; noise sigma=.01/.03 (seed fixed per image);
unsharp F+k(F-G1.5*F), k=1/2; gamma .7/1.4; contrast 1.25(F-mean)+mean; Gaussian blur sigma=1 (negative control).
Baselines: IR, VIS, pixel average. Chroma is always that of the visible image (BT.601 YCbCr), including for the
baselines and for methods that output RGB, so only Y differs between conditions.
Blending can restore source information the method lost. Noise, unsharp, gamma, contrast and blur are functions
of F alone and cannot add information about the sources (data-processing inequality); gamma is near invertible.

## Segmentation
Two public Cityscapes models never trained on fused images: SegFormer-B5 and Mask2Former Swin-S. Class map:
person={person,rider}, car={car,truck,bus}, bike={bicycle,motorcycle} against MSRS person, car, bike; every other
label counts as background. Score: dataset IoU per class and the mean of the three (mIoU3), overall, day and night.
Change against native: paired bootstrap over images, 2000 resamples, 95% CI. Input fixed at 2x (1280x960), near the
object scale the models were trained on; no other input size is evaluated for the headline result.

## Metrics
metric_full (13 metrics), Nabf sign flipped so higher is better. Change against native: Wilcoxon signed-rank over
images, Holm-corrected across the 13 metrics within each transformation.

## Humans
Two-alternative forced choice, native vs transformed, sources shown, left/right random. Comparisons: blend .25, blend
.5, average, noise .01, noise .03, unsharp 1, unsharp 2, gamma .7, contrast, blur. Three lists of 120 trials: 100 test,
10 catch (native vs sigma=.1 noise plus sigma=3 blur), 10 repeats with sides swapped. Target 18 raters, 6 per list.
Exclusion fixed in advance: catch accuracy below 80%. Primary: proportion preferring the transformed image per comparison, with
95% bootstrap CI clustered on rater and image, Holm-corrected tests against 0.5. Stop at the target; no looks at
outcomes to decide when to stop. If fewer than 15 raters remain after exclusion, report the study as underpowered.

## Decision rule for a metric m and transformation T
- Inflated: m rises significantly, and neither mIoU3 (both models) nor humans improve (CI upper bound of change <= 0 for
  mIoU3 in at least one model, and human CI upper bound <= .5 or containing .5).
- Supported: m rises significantly and humans prefer T (CI lower bound > .5).
- Missed degradation: T is blur or noise, humans prefer native, and m rises or does not fall.
The story follows from the table. If blends raise the metrics and humans prefer blends, the conclusion is
that deep methods over-process, not that the metrics are gamed. Both outcomes are reported.

## Addendum 2026-10-31, second pass (written after the four-method results, before any result of the two added methods)
Added methods: CDDFuse and SeAFusion, full MSRS test by their published weights, same conditions, models and metrics.
SeAFusion was trained with a segmentation loss, so it is the method most likely to beat the pixel average on this task.
The first-pass tables stay as they are. New tables pool six methods. New analysis, added because the four-method result
raised the question: for each method, native against the pixel average, mIoU3 difference with 95% paired bootstrap CI,
both models, all/day/night. The human study is not changed and does not include the two added methods.

## Addendum 2026-10-31, human study size and decision rule (written before any human response existed; responses/ was empty)
Task and metric results of both passes had been seen. This addendum replaces the matching text above; the rest stands.
Why: a simulation of this design (human_study/power_sim.py, results/human_power.csv, variances assumed) gave 18 raters on
3 lists about .47 power for a true preference of .65. The old rule also counted a human CI containing .5 as "humans do not
improve", so a small study would have labelled metrics inflated from lack of evidence.
- Size: 36 raters, 6 per list, 6 lists of 120 trials, codes R01-R36, trials per comparison and rater unchanged. Stop at 36.
  Excluded raters are not replaced. If fewer than 30 raters remain after exclusion, report the study as underpowered.
- Human outcome per comparison: prefers transformed (CI lower bound > .5), prefers native (CI upper bound < .5), or
  inconclusive (CI contains .5), the last reported with its width and never counted as evidence either way.
- Inflated: m rises significantly, mIoU3 CI upper bound <= 0 in at least one model, and humans prefer native.
- Supported: m rises significantly and humans prefer the transformed image.
- Missed degradation: T is blur or noise, humans prefer native, and m rises or does not fall.
- m rises significantly and humans are inconclusive: no verdict; the task result is reported alone.

## Addendum 2026-10-31, rank and class analyses of the existing data (written before any of these numbers was computed)
Pooled transformation effects, per-metric changes and native-vs-average differences had been seen. No new runs; the
tables above stay.
- Rank attack. Methods are ranked on the mean of a metric over the images of a dataset (Nabf reversed), the way comparison
  tables do it. One method at a time receives one transformation of F alone (noise, unsharp, gamma, contrast, blur), the
  other five stay native. Report its rank gain with a 95% paired bootstrap CI over images (5000 resamples); a gain counts
  if the lower bound is >= 1. Also report whether it reaches first place, and which metrics allow first place from any
  starting rank. Paired with the same method's own mIoU3 change under that transformation (MSRS only), the attack is
  called harmful if the gain counts and the mIoU3 CI upper bound is <= 0 in at least one model. Blends are not attacks,
  because they can restore source information, and are reported separately.
- Pixel average as a seventh entry, native table: its rank per metric and dataset, descriptive.
- Task class breakdown: person, car, bike IoU change per transformation, six methods pooled, 95% CI, both models.
- Multiplicity on the task: two-sided bootstrap p (10000 resamples), Holm over the 12 pooled transformation rows within a
  model, and Holm over the 6 native-vs-average differences within a model and scope. The uncorrected CIs stay reported.

## Addendum 2026-10-31, segmentation retrained on fused images (written before any model was trained; test set not used for any choice)
Why: pretrained Cityscapes models never saw fused images, so the pixel-average result may not hold for a model trained on
them, which is the common protocol in this field. Results with the pretrained models had been seen.
- Inputs (9): the six methods, pixel average, visible, infrared. Train on all 1083 MSRS train pairs with labels, test on the
  361 MSRS test pairs. No image content is shared between the splits (hash check on visible and infrared). Train fused images
  come from the same published weights and inference scripts as the test images. Luminance plus visible chroma, as everywhere.
  The fusion models may themselves have been trained on this split; that is usual and is reported.
- Model: SegFormer-B5, encoder and decoder from the Cityscapes weights already used, new 9-class classifier. One
  architecture, 3 seeds (0, 1, 2), 27 runs. The same seed gives every input the same order, crops and flips.
- Training, fixed: 30 epochs (8100 steps), batch 4, crop 384 (480 ran out of memory; found on training data only), random
  scale .75-1.5, flip, cross entropy, AdamW lr 6e-5 (classifier and decoder x10), weight decay .01, warmup 500 steps,
  linear decay. Last checkpoint only: no validation set, no model selection, no tuning after seeing test numbers.
- Test: full image at native scale. Scores: mIoU over the 9 MSRS classes (mIoU9) and over person, car, bike (mIoU3, also
  comparable in kind with the pretrained tables, not in value). Scores are averaged over seeds.
- Primary comparison: each method minus pixel average on mIoU9, and on mIoU3; 95% paired bootstrap CI over test images
  (10000 resamples), Holm over the 6 methods within score and scope; all, day, night. Seed-to-seed SD reported.
- Reading: if methods beat the average here, "deep methods are about equal to the pixel average" holds only for models that
  never saw fused images, and that is what gets reported. If they do not, the claim extends to this retraining protocol.
  One architecture and one dataset either way.

## Addendum 2026-11-02, checks added after the retraining results had been seen (exploratory, not part of the plan above)
retrain_sensitivity.py, results/retrain_sensitivity.json. These were chosen after seeing retrain_vs_avg.csv and are labelled
exploratory wherever reported. The planned comparison and its Holm correction stay the primary result.
- Images and seeds resampled together (10000 draws, method and reference share the seed draw), because the planned interval
  holds the three seeds fixed. With three seeds this is a rough widening, not an exact interval.
- Pixel average minus visible, and every input minus infrared. No multiplicity correction.
- Per-class IoU and class support. Guardrail appears in 10 test images, 3 at night, so mIoU9 is dominated by seed noise in
  that class; mIoU3 is the steadier score. No equivalence margin was fixed in advance, so no equivalence is claimed.

## Addendum 2026-11-02, transformed images on retrained models (written before any such model was tested)
Why: in the rank attack, "the task falls" rests on pretrained models only, and the retraining runs did not keep weights. Seen so
far: rank attack results on pretrained models, retraining results on native inputs. Nothing on retrained models with transformed input.
- Models: the six methods only, seed 0, same script and settings as the retraining addendum, each trained on that method's native
  train images, weights kept. No choice depends on test numbers. Native scores are compared with the earlier seed-0 runs as a
  repeatability check.
- Test: each method's own model on that method's native image and all 11 transformed versions (3 blends, 8 functions of F alone), 361
  images, full image at native scale.
- Primary score mIoU3, mIoU9 reported too. Change against native: 95% paired bootstrap over images, 10000 resamples. Six methods pooled:
  two-sided p, Holm over the 11 rows. Per method: CI only. Class breakdown person, car, bike, pooled.
- Rank attack join: the attacks counted in rank_attack.csv on MSRS are paired with the retrained change, using the same rule, harmful if the
  gain counts and the mIoU3 CI upper bound is <= 0. Also reported: counted attacks whose CI lower bound is > 0 (the task rises with the metric).
- Reading: if harm persists, "rank up while task falls" no longer rests on pretrained models alone. If it disappears, it is reported as
  specific to them. One architecture, one seed per method, and images the models were not trained on: a fall may reflect the shift
  between train and test appearance as well as lost information. Models trained on transformed images are not tested here.

## Addendum 2026-11-02, second labelled dataset FMB (written before any fused image, metric or segmentation result of FMB existed)
Seen on FMB so far: file lists, label value counts, a contact sheet of the training labels, split overlap checks. All MSRS results of the
sections above had been seen, so this is a replication with the design fixed in advance. Nothing on FMB is tuned.
- Data: official split, train 1220, test 280, 800x600, downloaded from the Drive folder in the SegMiF README. Label 0 is unlabelled (1.84% of
  pixels) and is excluded from predictions, labels and the loss. Bicycle has no test pixel and 12 training images, so 13 classes can be scored.
  The split interleaves frame numbers: 39 of 280 test images have a neighbouring frame in train, 2 are under 5 gray levels from a train image.
  This is reported. No public day or night tag exists, so only the full test set is scored; no scope is derived from brightness.
- Conditions and methods: the same six methods with the same weights and scripts, the same 11 transformations, baselines IR, VIS, pixel
  average. The catch condition is not rendered, since FMB is not in the human study. Fused train images for the six methods also.
- Pretrained segmentation: SegFormer-B5 and Mask2Former Swin-S, 3 classes person={8}, car={9,10,11}, bike={12,13}. Input fixed at 1280x960 (1.6x), the
  same model input size as on MSRS. Statistics as in the second pass plus the Holm rules of the rank addendum.
- Metrics: metric_full on 100 test images, seed 20261031, all conditions and baselines. Wilcoxon and Holm as before. No RoadScene-style pairing.
- Rank attack: the same 48 attacks per metric on these 100 images, same counting rule. Task change from each pretrained model and from the retrained model.
- Retraining: SegFormer-B5, same settings as the MSRS retraining (30 epochs, here 9150 steps; batch 4, crop 384, scale .75-1.5, flip, AdamW lr 6e-5,
  decode head x10, wd .01, warmup 500, linear decay), 14 classes, last checkpoint only, 9 inputs x seeds 0,1,2. At seed 0 the six methods also
  keep weights and are tested on all 11 transformations, as in the previous addendum. Scores: mIoU over 13 classes (mIoU13) and mIoU3 with the groups above.
- Primary: method minus average on mIoU13 and mIoU3, method minus visible, 95% paired bootstrap over images, Holm over 6; images and seeds resampled
  together is reported beside it and here is planned, not exploratory. Average minus visible and minus infrared, uncorrected. No equivalence is claimed.
- Reading: if methods beat the average on FMB, the MSRS claim does not generalise and the paper says so. The MSRS rank attack result counts as
  replicated for a metric when it has at least half as many counted attacks on FMB as on MSRS. Disagreements between datasets are reported, not resolved
  by choosing one. If a fusion script runs out of memory at 800x600, the image is run on the CPU with the same script and this is reported.

## Limits to report whatever the outcome
Cityscapes models are daylight models, and MSRS is partly at night. Task accuracy is one proxy, not image quality.
Only four methods in the first pass. Raters are not a population sample.

## Addendum 2026-11-16, the proposed network of the companion paper as a further input (written before any of its outputs on these splits existed)
Why: the two studies are merged into one manuscript, so the network proposed there is put through the same task test. Seen so far: every
result above, including FMB. Nothing about this network on the full MSRS test, MSRS train, FMB or the 30 RoadScene metric images.
- Model: the reported model of the companion paper (results/fusion_net_v3.mat, training seed 0, trained on 200 MSRS train pairs), unchanged.
  Fused luminance by fuse_images.m (per-image min-max scaling, as in that paper), saved as 8-bit PNG: MSRS test 361, MSRS train 1083, FMB test
  280, FMB train 1220, RoadScene 40. Luminance plus visible chroma, as everywhere. Native images only; no transformation, no rank attack.
- Metrics: metric_full on the native images of the metric sets (MSRS 100, RoadScene 30, FMB 100), reported descriptively beside the six methods.
- Pretrained segmentation: SegFormer-B5 and Mask2Former, native only, MSRS 361 and FMB 280, same settings.
- Retraining: same scripts and settings, seeds 0, 1, 2, on MSRS (9 classes) and FMB (14 classes); last checkpoint, no test-based choice.
- Comparisons: network minus pixel average and minus visible, mIoU3 and mIoU9/mIoU13, 95% paired bootstrap over images (10000) and images
  plus seeds; reported with CIs, not added to the Holm families of the six methods. Its position among the six methods is descriptive.
- Reading: if it does not beat the pixel average, the paper says so for its own network as for the others.
