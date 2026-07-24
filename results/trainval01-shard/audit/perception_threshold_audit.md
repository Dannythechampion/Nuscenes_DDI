# nuScenes Perception DDI Threshold Audit

## Scope

- Samples: 3,376
- Scenes: 85
- Unit of analysis: annotated 2 Hz keyframe
- Components: object complexity, visibility, LiDAR sparsity, camera truncation

## Composite score

- Mean: 61.61
- Standard deviation: 12.44
- Range: 0.00 to 80.55
- 25/50/75 percentiles: 57.25 / 62.93 / 70.64
- Fixed 0-33/33-67/67-100 split: Low 3.6%, High 37.3%

## Component saturation

| Component | Mean | P25 | P50 | P75 | Minimum-bin share | Maximum-bin share |
|---|---:|---:|---:|---:|---:|---:|
| count_score | 8.55 | 10.00 | 10.00 | 10.00 | 7.0% | 78.1% |
| visibility_mean_score | 3.50 | 2.47 | 3.47 | 4.53 | 3.6% | 0.3% |
| visibility_max_score | 9.39 | 10.00 | 10.00 | 10.00 | 3.6% | 91.0% |
| lidar_sparsity_mean_score | 7.11 | 6.43 | 7.32 | 8.18 | 0.9% | 0.9% |
| lidar_sparsity_max_score | 9.83 | 10.00 | 10.00 | 10.00 | 0.9% | 97.1% |
| truncation_mean_score | 0.33 | 0.00 | 0.16 | 0.47 | 37.4% | 0.1% |
| truncation_max_score | 4.41 | 0.00 | 3.00 | 10.00 | 37.4% | 27.2% |

## Interpretation rules

- A component with a large maximum-bin share is saturated and has weak ranking power.
- Empty or highly imbalanced Low/Medium/High groups indicate that fixed cutoffs do not match the observed distribution.
- Empirical tertiles are diagnostics, not final thresholds. Final thresholds require model-error validation on trainval.
- High correlation between components indicates duplicated evidence and should trigger a weighting review.

## Spearman correlations

| Feature | annotation_count | vru_count | vehicle_count | weighted_complexity | visibility_mean_score | lidar_sparsity_mean_score | truncation_mean_ratio | perception_difficulty_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| annotation_count | 1.000 | 0.539 | 0.611 | 0.927 | 0.306 | 0.175 | 0.227 | 0.658 |
| vru_count | 0.539 | 1.000 | -0.021 | 0.747 | 0.160 | 0.255 | 0.112 | 0.475 |
| vehicle_count | 0.611 | -0.021 | 1.000 | 0.486 | 0.243 | 0.014 | 0.194 | 0.404 |
| weighted_complexity | 0.927 | 0.747 | 0.486 | 1.000 | 0.314 | 0.255 | 0.181 | 0.642 |
| visibility_mean_score | 0.306 | 0.160 | 0.243 | 0.314 | 1.000 | 0.203 | 0.066 | 0.460 |
| lidar_sparsity_mean_score | 0.175 | 0.255 | 0.014 | 0.255 | 0.203 | 1.000 | -0.468 | 0.039 |
| truncation_mean_ratio | 0.227 | 0.112 | 0.194 | 0.181 | 0.066 | -0.468 | 1.000 | 0.655 |
| perception_difficulty_score | 0.658 | 0.475 | 0.404 | 0.642 | 0.460 | 0.039 | 0.655 | 1.000 |

## Next validation step

Run multiple perception models on the same keyframes, calculate per-object false-negative and localization errors, and test monotonic error growth across DDI groups while controlling for raw object count and distance.
