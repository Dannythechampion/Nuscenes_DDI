# 인지 모델 실행 계획

## 1. 제약조건

- 데이터: NuScenes `v1.0-trainval` keyframe-only
- 로컬 GPU: RTX 4060 Ti 8GB
- 로컬 실행 환경: WSL Ubuntu 22.04, Python 3.10
- WSL virtualenv: `/home/hansol/venvs/nuscenes-ddi`
- Intermediate LiDAR sweeps와 temporal camera frame은 없음

Keyframe-only 입력에서 얻은 결과를 multi-sweep 또는 temporal 공식 checkpoint 성능과 동일한 조건으로 주장하면 안 된다.

## 2. 1차 모델 구성

### Camera-only

다음 모델은 single camera keyframe을 처리한 뒤 multi-view 결과를 결합하는 기준선으로 우선 검토한다.

1. FCOS3D
2. PGD

목표는 최고 성능 달성이 아니라, 구조가 다른 두 인지 모델에서 `P_camera`와 오류 증가 관계가 반복되는지 확인하는 것이다. 로컬 8GB GPU에서는 batch size 1과 mixed precision을 사용한다.

### LiDAR-only

공식 NuScenes PointPillars와 CenterPoint 설정은 일반적으로 과거 sweep을 누적한다. keyframe-only에서 기존 multi-sweep checkpoint를 그대로 실행하면 입력 분포가 달라져 공식 성능과 비교할 수 없다.

LiDAR 실험은 다음 중 하나로 진행한다.

1. `sweeps_num=0` 또는 single-sweep 설정으로 PointPillars를 train split에서 재학습한다.
2. single-sweep NuScenes 학습 조건이 명확히 공개된 checkpoint를 사용한다.
3. 일부 validation scenes의 공식 sweeps를 추가 확보해 multi-sweep 결과를 보조 실험으로 비교한다.

Single-sweep 재학습은 로컬 8GB보다 cloud GPU가 적합하다.

### Fusion

Fusion 모델은 Camera와 LiDAR 실험이 완료된 뒤 선택한다. 이 단계에서는 `P_camera`, `P_lidar`, `P_fusion` 중 어떤 지표가 오류를 가장 잘 설명하는지 비교한다.

## 3. 공정한 비교 조건

- 동일한 NuScenes validation keyframe을 사용한다.
- 공식 `detection_cvpr_2019` class range와 zero-sensor-point 필터를 적용한다.
- 모델 출력은 공식 NuScenes detection JSON 형식으로 저장한다.
- 각 모델에서 FN rate, FP/keyframe, translation/scale/orientation error를 계산한다.
- Camera 모델에는 `P_camera`, LiDAR 모델에는 `P_lidar`를 우선 적용한다.
- 공통 합성 DDI는 두 modality에서 독립적으로 효과가 확인된 이후에만 평가한다.
- 모델별 confidence threshold가 결론을 바꾸지 않는지 sensitivity analysis를 수행한다.

## 4. 로컬과 클라우드 역할

### 로컬 RTX 4060 Ti 8GB

- Mini end-to-end 검증
- Camera model batch-1 inference
- LiDAR single-frame inference
- prediction/GT matching
- 통계 분석 및 그래프 생성

### Cloud GPU

- Single-sweep LiDAR model 학습
- 6-view 대형 Camera model 추론
- Fusion model 학습 또는 추론
- 반복 실험과 threshold sensitivity run

## 5. 실행 순서

1. Trainval keyframe package의 완전성을 검사한다.
2. MMDetection3D용 train/validation info 파일을 만든다.
3. Mini에서 FCOS3D 또는 PGD prediction JSON을 생성한다.
4. `match_nuscenes_detection_errors.py`로 객체 오류표를 만든다.
5. Full validation split에서 Camera 모델 2개를 실행한다.
6. Cloud에서 single-sweep LiDAR baseline을 학습·평가한다.
7. 모델별 DDI-오류 관계와 단순 baseline을 비교한다.

## 6. 참고

- [MMDetection3D NuScenes dataset guide](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/datasets/nuscenes.md)
- [NuScenes object detection protocol](https://www.nuscenes.org/object-detection)
