# Perception DDI v1 Preregistered Validation Protocol

## 1. Research question

Do perception difficulty indicators transferred from established autonomous-driving
benchmarks predict object-detection errors on the official 150-scene nuScenes
validation split across at least two nuScenes-trained models?

External datasets are not used as experimental cohorts. Their published criteria
are translated into nuScenes variables and evaluated only on nuScenes.

## 2. Frozen validation cohort

- Dataset: nuScenes `v1.0-trainval`
- Split: official `val`
- Scenes: 150
- Keyframes: 6,019
- Camera observations per monocular model: 36,114 (`6,019 x 6`)
- Camera inputs: six official keyframe images
- LiDAR inputs: current keyframe plus up to ten preceding `LIDAR_TOP` sweeps;
  PointPillars consumes 10 and CenterPoint consumes 9 according to their configs
- Ground-truth eligibility: nuScenes detection classes, class ranges, and nonzero
  LiDAR-plus-radar point rule

No validation error may be used to alter the rules below. Any later fitted score
must be labeled exploratory and evaluated in a separate held-out cohort.

## 3. External criteria translated to nuScenes

### 3.1 Density anchor

nuPlan defines `near_multiple_vehicles` as more than six moving vehicles within
8 m while ego speed exceeds 6 m/s. The nuScenes translation uses annotation
attributes, ego-to-object distance, and ego speed estimated from adjacent
keyframe poses.

- `moving_vehicle_count_8m`: count of `vehicle.moving` annotations within 8 m
- `nuplan_near_multiple_vehicles`: count > 6 and estimated ego speed > 6 m/s
- `density_prior_score`: `min(10, count / 7 * 10)`

The earlier `2*VRU + vehicle + 0.5*static` expression remains an exploratory team
hypothesis. H3D supports the relevance of crowded scenes but does not establish
those weights or a 40-object threshold.

### 3.2 Camera visibility

nuScenes officially bins visibility over all six cameras into `0-40`, `40-60`,
`60-80`, and `80-100` percent. The continuous score uses hidden-fraction
midpoints: 8, 5, 3, and 1 on a 0-10 scale.

The ordinal rule is frozen as:

- Easy: `80-100%` visible
- Moderate: `40-80%` visible
- Hard: `0-40%` visible

### 3.3 Camera truncation

KITTI's 15%, 30%, and 50% truncation boundaries are applied to the ratio of the
projected nuScenes 3D box falling outside its best camera image.

- Easy: truncation <= 15%
- Moderate: 15% < truncation <= 30%
- Hard: truncation > 30%
- Values above 50% are retained as Hard rather than excluded, because the study
  predicts error instead of reproducing KITTI's benchmark inclusion policy.

Pixel-height thresholds are not transferred because KITTI and nuScenes camera
geometry and resolution differ. The percentage truncation criterion is directly
dimensionless; pixel height is not.

### 3.4 LiDAR sparsity

Waymo's five-point LEVEL boundary is transferred to nuScenes `num_lidar_pts`.
The zero-return case is separated because the current LiDAR keyframe contributes
no object return even when nuScenes retains the annotation due to radar points.
Historical sweeps may still contain past returns, so this is explicitly a
current-frame sparsity label rather than proof of zero temporal evidence.

- Easy: more than 5 LiDAR points
- Moderate: 1-5 LiDAR points
- Hard: 0 LiDAR points

## 4. Object and frame aggregation

Camera object difficulty is the worse ordinal level of visibility and truncation.
LiDAR object difficulty is the point-count level above.

Frame difficulty is the ceiling of the 75th percentile of eligible object levels.
A frame satisfying the translated nuPlan density event is Hard. This aggregation
avoids the saturation caused by a maximum while preserving the benchmark rule
that a difficult component cannot be canceled by an easy component.

Continuous scores are retained for association tests:

- `camera_ddi_prior_v1`: equal mean of density score, eligible-object visibility
  P75 score, and eligible-object truncation P75 score
- `lidar_ddi_prior_v1`: equal mean of density score and the proportion of eligible
  objects with at most five LiDAR points multiplied by 10

Equal weights are team-defined and therefore tested with component ablations.
Ordinal Easy/Moderate/Hard is the primary externally anchored classification.

### 4.1 Exploratory team density hypothesis

The presentation's `2*VRU + vehicle + 0.5*static` complexity and 40-object
boundary are retained as exploratory hypotheses, not treated as published H3D
thresholds. Their distributions and associations with frame error are reported
separately. A scene-clustered, GT-count-weighted frame regression estimates the
marginal VRU, vehicle, and static-object coefficients while controlling for mean
object distance and location. This directly tests whether the proposed 2:1:0.5
ordering is reflected in model errors without rewriting the weights from val.

## 5. Models

Camera validation:

- FCOS3D official MMDetection3D nuScenes fine-tuned checkpoint
- PGD official MMDetection3D nuScenes fine-tuned checkpoint

LiDAR validation:

- PointPillars official nuScenes checkpoint
- CenterPoint official nuScenes checkpoint

All checkpoints must be trained for nuScenes. KITTI, Waymo, H3D, and nuPlan
models are not run.

## 6. Primary outcomes and tests

- Object false-negative indicator
- Frame false-negative rate
- False positives per keyframe normalized by GT opportunities
- Matched-box translation error
- Spearman association with continuous modality DDI
- Scene-clustered binomial GLM controlling for class, distance, object count,
  box volume, location, and model
- Scene-resampled 95% bootstrap intervals for Easy/Moderate/Hard
- Object-FN AUC against distance, raw count, and inverse LiDAR points
- Threshold sensitivity for matching distance 1/2/4 m and confidence 0.05/0.10/0.20

## 7. Decision rule

A modality DDI is supported only if:

1. error generally increases from Easy to Moderate to Hard in both models;
2. the adjusted continuous DDI effect is positive in both models;
3. direction is stable under matching sensitivity analysis;
4. results are not explained solely by distance, class, location, or one scene;
5. the composite adds information beyond its strongest single component.

Failure to populate all three groups is itself evidence that the transferred rule
does not discriminate nuScenes and must be reported. It is not repaired by fitting
validation quantiles.

## 8. Primary sources

- nuScenes visibility and schema: https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html
- nuScenes detection protocol: https://www.nuscenes.org/object-detection
- KITTI object benchmark: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- nuPlan scenario definitions: https://nuplan-devkit.readthedocs.io/
- Waymo Open Dataset perception paper: https://arxiv.org/abs/1912.04838
- H3D crowded-scene dataset: https://usa.honda-ri.com/h3d
