# nuScenes 150-scene LIDAR DDI v1 validation

## Cohort

- 150 official validation scenes, 6,019 keyframes
- Models: centerpoint, pointpillars
- Matching runs: 10

## Preregistered group distribution

- Easy: 1,467 keyframes across 137 scenes
- Moderate: 4,477 keyframes across 150 scenes
- Hard: 75 keyframes across 26 scenes

## Primary model results

- centerpoint: Easy 0.0555, Moderate 0.0806, Hard 0.2508; strict monotonic=True
- pointpillars: Easy 0.0838, Moderate 0.1445, Hard 0.2821; strict monotonic=True

## Adjusted effects

- centerpoint frame_ddi: OR/point=1.0047 (95% CI 0.8884-1.1363, p=0.9399)
- centerpoint object_ddi: OR/point=0.7599 (95% CI 0.7325-0.7884, p=1.603e-48)
- pointpillars frame_ddi: OR/point=1.0803 (95% CI 0.9667-1.2072, p=0.1733)
- pointpillars object_ddi: OR/point=0.7667 (95% CI 0.7440-0.7901, p=3.507e-67)

## Frozen decision

- All three groups populated: True
- Strict Easy < Moderate < Hard in every primary model: True
- Positive significant adjusted frame-DDI effect in every model: False
- Positive significant adjusted object-DDI effect in every model: False
- Monotonic in at least 80% of sensitivity runs: True
- Composite stronger than every single component: True
- Overall preregistered support: **False**

A negative result is retained as evidence against the transferred rule; validation quantiles are not fitted.
