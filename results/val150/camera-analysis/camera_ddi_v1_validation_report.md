# nuScenes 150-scene CAMERA DDI v1 validation

## Cohort

- 150 official validation scenes, 6,019 keyframes
- Models: FCOS3D, PGD
- Matching runs: 10

## Preregistered group distribution

- Easy: 870 keyframes across 93 scenes
- Moderate: 2,680 keyframes across 145 scenes
- Hard: 2,469 keyframes across 143 scenes

## Primary model results

- FCOS3D: Easy 0.2138, Moderate 0.2158, Hard 0.2574; strict monotonic=True
- PGD: Easy 0.2543, Moderate 0.2895, Hard 0.3480; strict monotonic=True

## Adjusted effects

- FCOS3D frame_ddi: OR/point=1.0018 (95% CI 0.9070-1.1065, p=0.9711)
- FCOS3D object_ddi: OR/point=0.6971 (95% CI 0.6721-0.7230, p=1.114e-83)
- PGD frame_ddi: OR/point=1.0009 (95% CI 0.9178-1.0915, p=0.9839)
- PGD object_ddi: OR/point=0.6015 (95% CI 0.5780-0.6259, p=1.541e-138)

## Frozen decision

- All three groups populated: True
- Strict Easy < Moderate < Hard in every primary model: True
- Positive significant adjusted frame-DDI effect in every model: False
- Positive significant adjusted object-DDI effect in every model: False
- Monotonic in at least 80% of sensitivity runs: True
- Composite stronger than every single component: False
- Overall preregistered support: **False**

A negative result is retained as evidence against the transferred rule; validation quantiles are not fitted.
