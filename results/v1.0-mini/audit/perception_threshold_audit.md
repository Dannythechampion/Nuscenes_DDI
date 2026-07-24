# nuScenes Perception DDI Threshold Audit

## Scope

- Samples: 404
- Scenes: 10
- Unit of analysis: annotated 2 Hz keyframe
- Components: object complexity, visibility, LiDAR sparsity, camera truncation

## Composite score

- Mean: 65.48
- Standard deviation: 7.32
- Range: 35.84 to 78.62
- 25/50/75 percentiles: 59.59 / 66.22 / 70.82
- Fixed 0-33/33-67/67-100 split: Low 0.0%, High 48.8%

## Component saturation

| Component | Mean | P25 | P50 | P75 | Minimum-bin share | Maximum-bin share |
|---|---:|---:|---:|---:|---:|---:|
| count_score | 9.60 | 10.00 | 10.00 | 10.00 | 7.9% | 92.1% |
| visibility_mean_score | 3.42 | 2.36 | 3.44 | 4.40 | 0.2% | 0.2% |
| visibility_max_score | 9.73 | 10.00 | 10.00 | 10.00 | 2.7% | 94.6% |
| lidar_sparsity_mean_score | 6.96 | 6.36 | 6.93 | 7.71 | 0.2% | 0.2% |
| lidar_sparsity_max_score | 10.00 | 10.00 | 10.00 | 10.00 | 100.0% | 100.0% |
| truncation_mean_score | 0.30 | 0.09 | 0.23 | 0.38 | 19.8% | 0.2% |
| truncation_max_score | 5.82 | 3.00 | 7.00 | 10.00 | 19.8% | 36.9% |

## Interpretation rules

- A component with a large maximum-bin share is saturated and has weak ranking power.
- Empty or highly imbalanced Low/Medium/High groups indicate that fixed cutoffs do not match the observed distribution.
- Empirical tertiles are diagnostics, not final thresholds. Final thresholds require model-error validation on trainval.
- High correlation between components indicates duplicated evidence and should trigger a weighting review.

## Spearman correlations

| Feature | annotation_count | vru_count | vehicle_count | weighted_complexity | visibility_mean_score | lidar_sparsity_mean_score | truncation_mean_ratio | perception_difficulty_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| annotation_count | 1.000 | 0.708 | 0.608 | 0.960 | 0.635 | 0.450 | -0.099 | 0.624 |
| vru_count | 0.708 | 1.000 | 0.051 | 0.848 | 0.266 | 0.446 | -0.237 | 0.331 |
| vehicle_count | 0.608 | 0.051 | 1.000 | 0.481 | 0.591 | 0.103 | 0.038 | 0.499 |
| weighted_complexity | 0.960 | 0.848 | 0.481 | 1.000 | 0.575 | 0.490 | -0.167 | 0.556 |
| visibility_mean_score | 0.635 | 0.266 | 0.591 | 0.575 | 1.000 | 0.344 | -0.116 | 0.554 |
| lidar_sparsity_mean_score | 0.450 | 0.446 | 0.103 | 0.490 | 0.344 | 1.000 | -0.423 | 0.253 |
| truncation_mean_ratio | -0.099 | -0.237 | 0.038 | -0.167 | -0.116 | -0.423 | 1.000 | 0.474 |
| perception_difficulty_score | 0.624 | 0.331 | 0.499 | 0.556 | 0.554 | 0.253 | 0.474 | 1.000 |

## Next validation step

Run multiple perception models on the same keyframes, calculate per-object false-negative and localization errors, and test monotonic error growth across DDI groups while controlling for raw object count and distance.
