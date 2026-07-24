# NuScenes Perception DDI Validation Plan

## 1. Research question

Does the proposed perception difficulty index predict higher object-detection error on nuScenes keyframes?

The DDI is a hypothesis, not a ground-truth label. Its validity must be tested against errors from multiple perception models. The claim is limited to annotated 2 Hz keyframes unless temporal models and intermediate sweeps are added later.

## 2. Dataset scope

- Development check: official `v1.0-mini`, 10 scenes and 404 keyframes.
- Main analysis: `v1.0-trainval`, 850 scenes and 34,149 keyframes.
- Required keyframe data: six cameras, `LIDAR_TOP`, metadata, annotations, calibration, ego pose, and maps.
- Intermediate sweeps are excluded from the first experiment.
- Train scenes are used for threshold development; validation scenes are held out for final reporting.

## 3. Candidate perception variables

| Variable | Current proposal | Evidence status | Validation treatment |
|---|---|---|---|
| Object complexity | Weighted count with cutoffs 5 and 15; raw count 40 | No official KITTI or nuScenes cutoff | Compare fixed cutoffs with train-set quantiles and continuous count. Control for raw object count when testing error rates. |
| Visibility | nuScenes tokens 1-4 mapped to 10/7/3/0 | Token intervals are official nuScenes metadata | Keep token intervals, but replace frame maximum with proportions in each visibility bin and mean severity. |
| Truncation | 0.05/0.30/0.50 | KITTI uses maximum truncation 0.15/0.30/0.50 for Easy/Moderate/Hard; nuScenes has no official truncation difficulty label | Treat both cutoff sets as candidates. Report that camera projection is a derived proxy, not an official nuScenes label. |
| LiDAR sparsity | 50/20/5 points | `num_lidar_pts` is official metadata; these thresholds are not an official difficulty standard | Use log point count, class-distance normalization, and train-set quantiles. Avoid maximum severity because one zero-point object saturates the frame. |

The official nuScenes evaluation removes a ground-truth box only when the sum of LiDAR and radar points is zero. It does not remove every box with zero LiDAR points. Point-count filtering and DDI analysis must therefore use explicit modality labels.

The primary result should report a shared scene-context vector plus modality-conditioned scores:

- `P_camera`: visibility, projected size/truncation, class, distance, and lighting.
- `P_lidar`: LiDAR point count normalized by class and distance, class, distance, and crowding.
- `P_fusion`: camera and range-sensor components, with missing-sensor interactions tested explicitly.

A single blended score may be reported only after showing that its relationship with error is stable across sensor modalities.

The current composite averages one count component and two components each for visibility, LiDAR sparsity, and truncation. This creates an implicit 1:2:2:2 domain weighting before any evidence-based weights have been justified. The existing output is therefore labeled `presentation_legacy_v0` and is retained only as a baseline. A revised score must aggregate each domain once or estimate weights using train-only model-error data.

## 4. Mini audit findings

The first audit on 404 keyframes found substantial saturation:

- Object-count score at its maximum: 92.1% of keyframes.
- Maximum visibility score at its maximum: 94.6%.
- Maximum LiDAR sparsity score at its maximum: 100.0%.
- A fixed composite split of 0-33/33-67/67-100 produced no Low samples.

These results do not invalidate the underlying variables. They show that the current aggregation and thresholds have weak discrimination and must be calibrated on trainval before they are used as final DDI rules.

## 5. Model-error experiment

Use at least two models with different sensor modalities if the available hardware permits:

1. A single-keyframe LiDAR detector.
2. A camera-only detector using nuScenes camera keyframes.
3. Optionally, a fusion detector for a robustness check.

Temporal or multi-sweep models belong in a separate experiment because keyframe-only input changes their official inference setting.

For every keyframe, match predictions to ground-truth boxes using the official nuScenes center-distance thresholds. Record:

- False-negative rate per ground-truth object.
- False positives per keyframe.
- Translation, scale, orientation, velocity, and attribute errors for matched boxes.
- Class-specific and distance-specific recall.
- Official mAP and NDS components for aggregate reporting.

Raw error count is not the primary target because crowded scenes contain more opportunities for errors. The main target is per-object error probability, with false positives reported per frame.

## 6. Statistical validation

1. Fit thresholds on the train split only.
2. Measure Spearman correlation between each continuous feature and model error.
3. Compare Low/Medium/High groups using scene-level bootstrap confidence intervals.
4. Fit an object-level error model controlling for class, distance, box size, city, and model.
5. Test whether error increases monotonically across DDI groups.
6. Compare DDI against simple baselines: raw object count, median distance, minimum LiDAR points, and visibility alone.
7. Vary every threshold and weight to test sensitivity.
8. Confirm results separately by model, class, city, day/night condition, and train/validation split.

## 7. Acceptance criteria

The perception DDI is supported only if all of the following hold:

- Per-object error increases monotonically from Low to High.
- The relationship remains after controlling for object count, class, and distance.
- The effect appears in at least two perception models.
- DDI predicts error better than simple one-variable baselines.
- Small threshold changes do not reverse the conclusion.
- Validation-split results agree with train-split development results.

If these conditions fail, report which variables remain useful and revise or remove the saturated components.

## 8. Execution order

1. Validate the keyframe-only trainval package for file and annotation completeness.
2. Extract continuous perception features for all trainval keyframes.
3. Audit distributions, missingness, saturation, and correlations.
4. Define candidate thresholds using literature values and train-only empirical quantiles.
5. Run perception models and save official-format prediction JSON files.
6. Build object-level matched-error tables.
7. Run statistical and baseline comparisons.
8. Freeze thresholds and report held-out validation results.

## 9. Primary references

- KITTI object benchmark difficulty criteria: https://www.cvlibs.net/datasets/kitti/eval_object.php
- nuScenes schema and visibility definitions: https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html
- nuScenes detection metrics and challenge protocol: https://www.nuscenes.org/object-detection
- nuScenes paper: https://arxiv.org/abs/1903.11027
