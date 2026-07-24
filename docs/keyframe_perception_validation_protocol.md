# Keyframe Perception DDI Validation Protocol

## Scope

- Dataset package: nuScenes `v1.0-trainval` trainval01 keyframe shard
- Complete cohort: 3,376 keyframes from 85 scenes
- Threshold-development split: 2,462 official train keyframes from 62 scenes
- Held-out model-test split: 914 official validation keyframes from 23 scenes
- Camera inference observations: 5,484 (`914 x 6 cameras`)
- Intermediate camera frames and LiDAR sweeps: excluded

This is not the complete 34,149-keyframe nuScenes trainval release. Results must be
reported as a validation on the available complete-scene shard.

## Models

| Model | Input | Official config | Checkpoint SHA-256 |
|---|---|---|---|
| FCOS3D | Monocular camera, six views merged by sample token | `fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py` | `8d806dc2ecae85bc8eaba1f16dccdf03459317ca6aed7984cfda33c2a2bc33a8` |
| PGD | Monocular camera, six views merged by sample token | `pgd_r101-caffe_fpn_head-gn_16xb2-1x_nus-mono3d_finetune.py` | `fd419681dece0fc17a5fe4a5eaa0fbd334eb77d7f95b9f8fa7babf02d8b32efe` |

- MMDetection3D: `v1.4.0`, commit `fe25f7a51d36e3702f961e198894580d83c4387b`
- PyTorch: `2.1.2+cu121`
- GPU: NVIDIA GeForce RTX 4060 Ti 8GB
- Batch size: 1

The two models provide model replication within the camera modality. They do
not constitute an independent validation of the LiDAR sparsity component.

## Leakage Control

1. Compute the existing `presentation_legacy_v0` indicators for every complete keyframe.
2. Derive provisional Easy/Moderate/Hard boundaries from train DDI tertiles only.
3. Do not use model errors to move those boundaries.
4. Run both pretrained models only on the held-out official validation scenes.
5. Retain the presentation's fixed `33.33/66.67` split as a non-fitted baseline.

Train tertiles are operational candidate boundaries, not literature-backed
ground truth. They test rank separation without fitting to validation errors.

## Error Matching

Ground-truth eligibility follows the nuScenes detection class ranges and removes
annotations only when LiDAR plus radar points are zero. Predictions are processed
in descending confidence order and matched one-to-one to the nearest unmatched
ground-truth object of the same class.

- Primary center-distance threshold: 2m
- Distance sensitivity: 1m and 4m
- Primary confidence threshold: 0.05
- Confidence sensitivity: 0.10 and 0.20
- Primary targets: per-object false negative, false positives per keyframe,
  and translation error for matched boxes

Raw error count is not a primary outcome because dense scenes create more error
opportunities.

## Statistical Tests

- Spearman association between continuous DDI/components and frame error
- Easy/Moderate/Hard error rates with scene-resampled bootstrap confidence intervals
- Object-level binomial GLM with scene-clustered standard errors
- GLM controls: class, object distance, raw object count, box volume, and location
- AUC comparison against raw object count, distance, and inverse LiDAR point count
- Separate reporting by model and matching sensitivity condition

The proposed three-level DDI is supported only when error increases monotonically
across all three groups in both models, the adjusted DDI effect is positive, DDI
outperforms simple baselines, and sensitivity runs do not reverse the conclusion.

## Reproduction Entry Points

- `scripts/build_keyframe_eval_manifest.py`
- `scripts/prepare_mmdet3d_keyframe_infos.py`
- `configs/fcos3d_keyframe_val.py`
- `configs/pgd_keyframe_val.py`
- `scripts/match_nuscenes_detection_errors.py`
- `scripts/analyze_perception_model_validation.py`

## Primary References

- [MMDetection3D nuScenes dataset guide](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/datasets/nuscenes.md)
- [MMDetection3D model repository](https://github.com/open-mmlab/mmdetection3d)
- [nuScenes detection task and metrics](https://www.nuscenes.org/object-detection)
- [nuScenes paper](https://arxiv.org/abs/1903.11027)
