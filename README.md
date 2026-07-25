# NuScenes DDI Perception Research

NuScenes 장면의 인지 난이도 지표(DDI)가 실제 자율주행 객체 검출 오류 증가를 설명하는지 검증하는 프로젝트입니다.

## 연구 질문

> 객체 밀집도, 가림, 센서 희소성, 화면 잘림과 같은 장면 특성이 높을수록 서로 다른 자율주행 인지 모델이 실제로 더 자주 틀리는가?

단순히 난이도 점수를 만드는 것이 아니라, 각 기준값의 출처와 통계적 타당성을 제시하고 모델의 FN/FP 및 위치·크기·방향 오차와 연결하는 것이 목표입니다.

전체 연구 목적과 검증 절차는 [DDI 연구 목적 및 타당성 검증 프레임워크](docs/research_objective_and_validation_framework.md)에 정리되어 있습니다.

## 현재 진행 상태

| 단계 | 상태 | 산출물 |
|---|---:|---|
| 공식 `v1.0-mini` 완전성 검사 | 완료 | 10 scenes, 404 keyframes, 18,538 annotations |
| 인지 특징 추출 | 완료 | keyframe/object/scene 단위 특징표 |
| 기존 임계값 감사 | 완료 | 포화율, 분포, 상관관계, 클래스·거리별 LiDAR 분석 |
| prediction-GT 오류 매칭 | 완료 | perfect prediction self-test에서 FN=0, FP=0 |
| trainval01 shard 검증 | 완료 | 85 scenes, 3,376 keyframes, 전체의 9.886% |
| 공식 val 150-scene keyframe 확보 | 완료 | 6,019 keyframes, camera/LiDAR 누락 0 |
| LiDAR multi-sweep 확보 | 완료 | 요구 파일 57,916개, 누락 0 |
| Camera 모델 추론 | 완료 | FCOS3D, PGD |
| LiDAR 모델 추론 | 완료 | PointPillars, CenterPoint |
| DDI-오류 관계 검정 | 완료 | 4 models, 20 sensitivity runs |
| 인지 DDI v1 최종 판정 | 완료 | 사전등록 전체 지지 실패 |

## 150-scene 본실험 결론

공식 validation 150개 scene 전체에서 Camera와 LiDAR DDI의 Easy, Moderate, Hard는 네 모델 모두에서 원시 FN rate가 단조 증가했다. 그러나 거리·클래스·객체 수·box 크기·장소와 scene 군집을 통제한 DDI 효과는 모든 모델에서 유의한 양의 효과가 아니었다.

따라서 **현재 합성 DDI v1은 확정 지표로 채택하지 않는다.** 가림과 LiDAR 5-point 희소성은 유망한 component로 남지만, 밀집도 가중치, 동일 가중 평균, frame P75 집계는 train 또는 별도 개발 cohort에서 재설계한 뒤 새로운 held-out 자료로 검증해야 한다.

- [150-scene 전체 검증 보고서](results/val150/인지_DDI_150scene_검증_보고서.md)
- [사전등록 검증 절차](docs/perception_ddi_v1_preregistered_protocol.md)
- [모델 성능 요약](results/val150/model_performance_summary.csv)
- [최종 판정 요약](results/val150/validation_decision_summary.csv)

## 1차 분석 결론

Mini와 trainval01 shard 분석은 현재 발표 기준을 그대로 확정 임계값으로 사용하기 어렵다는 것을 보여줍니다.

- 객체별 LiDAR point가 `0-4`인 annotation이 약 `53.5%`였습니다.
- LiDAR point 수는 객체 거리와 클래스에 강하게 종속됩니다.
- 기존 component max 방식은 일부 항목에서 `92-100%`가 최고 난이도로 포화됐습니다.
- 기존 합성 점수는 component별 값 범위 차이로 사실상 `1:2:2:2` 가중치를 갖습니다.
- 고정 점수 구간은 Low 장면을 만들지 못하고 Medium/High로만 분리됐습니다.

따라서 DDI는 하나의 공통 점수부터 만들기보다 다음처럼 센서 조건별로 검증합니다.

- `P_camera`: 가림, truncation, 투영 크기, 조도·날씨
- `P_lidar`: 클래스·거리 조건부 point 희소성, 가림, 밀집도
- `P_fusion`: Camera/LiDAR 요인의 결합 및 상호작용

위 항목은 초기 `v1.0-mini` 진단 결과입니다. 이후 공식 val 150개 scene과 네 모델로 수행한 본실험 결과는 위 150-scene 보고서를 기준으로 봅니다.

## 저장소 구성

```text
docs/
  perception_research_plan.md          연구 설계와 통계 검증 계획
  perception_model_execution_plan.md   Camera/LiDAR/Fusion 실행 계획
  perception_threshold_evidence.csv    기준값 근거 정리
  인지_DDI_1차_분석_보고서.md           Mini 분석 결과 보고서
results/v1.0-mini/
  audit/                                핵심 요약 CSV와 감사 보고서
  figures/                              포화도·점수·LiDAR 분포 그래프
  keyframe_validation.json              데이터 완전성 검사 결과
scripts/
  validate_nuscenes_keyframes.py        keyframe 파일/metadata 완전성 검사
  extract_nuscenes_perception_features.py
  audit_perception_thresholds.py
  plot_perception_audit.py
  match_nuscenes_detection_errors.py    NuScenes prediction과 GT 오류 매칭
  materialize_nuscenes_keyframes.ps1    Drive 원본 로컬화·압축해제·검증
  stream_copy_file.py                   대용량 Drive 파일 스트리밍 복사·재개
  resume_range_downloads.py             HTTP byte-range 다운로드 재개 유틸리티
```

## 환경 구성

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .venv-analysis
.\.venv-analysis\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
```

## Mini 분석 재현

공식 NuScenes `v1.0-mini`를 `data/nuscenes`에 배치한 후 실행합니다.

```powershell
python scripts\validate_nuscenes_keyframes.py `
  --dataroot data\nuscenes `
  --version v1.0-mini `
  --output outputs\perception\v1.0-mini_keyframe_validation.json

python scripts\extract_nuscenes_perception_features.py `
  --dataroot data\nuscenes `
  --version v1.0-mini `
  --output-dir outputs\perception

python scripts\audit_perception_thresholds.py `
  --samples-csv outputs\perception\v1.0-mini_perception_samples.csv `
  --objects-csv outputs\perception\v1.0-mini_perception_objects.csv `
  --output-dir outputs\perception\audit-mini

python scripts\plot_perception_audit.py `
  --samples-csv outputs\perception\v1.0-mini_perception_samples.csv `
  --component-audit-csv outputs\perception\audit-mini\component_threshold_audit.csv `
  --object-threshold-csv outputs\perception\audit-mini\object_threshold_distribution.csv `
  --output-dir outputs\perception\audit-mini\figures
```

## Trainval 데이터 사용

다운로드한 Kaggle package를 실제 파일 단위로 검증한 결과, 전체 keyframe-only trainval이 아니라 공식 `v1.0-trainval01_blobs.tgz`와 전체 metadata를 묶은 단일 shard였습니다.

- 압축 크기: `33,489,285,238 bytes` (`31.19 GiB`)
- 포함 내용: 공식 trainval metadata, `trainval01` samples와 sweeps
- 완전 keyframe: `3,376 / 34,149` (`9.886%`)
- 완전 scene: `85 / 850` (`train 62`, `val 23`)
- 누락 keyframe: `30,773`

이 shard는 pilot 분석과 23개 validation scene 평가에는 사용할 수 있지만, 전체 trainval 결과로 보고하면 안 됩니다. 전체 keyframe-only 구성을 만들려면 공식 blob 10개를 모두 받은 뒤 각 archive에서 `samples`만 병합하고 `sweeps`를 제거하거나, 34,149개 keyframe 완전성을 검증할 수 있는 별도 mirror가 필요합니다.

Google Drive에서 shard를 다시 로컬화할 때는 다음 명령을 사용합니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\materialize_nuscenes_keyframes.ps1 `
  -Extract `
  -Validate
```

검증기는 `850 scenes`, `34,149 keyframes`, 6개 camera channel, `LIDAR_TOP`, annotation/calibration/ego pose/map 및 모든 keyframe 파일 경로를 확인합니다. 일부 shard라면 전체 통과 대신 complete sample/scene coverage를 보고합니다.

## 본실험 모델 검증 절차

1. 외부 기준을 nuScenes 변수로 번역하고 validation 결과 확인 전에 고정했습니다.
2. Camera FCOS3D·PGD와 LiDAR PointPillars·CenterPoint를 공식 val 6,019개 keyframe에서 실행했습니다.
3. LiDAR 모델에는 공식 config대로 과거 9-10개 sweep을 제공했습니다.
4. 예측과 GT를 동일 클래스 중심거리로 매칭해 FN, FP, translation error를 계산했습니다.
5. Scene bootstrap, scene-clustered 회귀, 단일 component·기준선 비교, threshold 민감도 분석을 수행했습니다.
6. 사전등록 조건을 하나라도 충족하지 못하면 DDI 전체를 지지하지 않는 것으로 판정했습니다.

## 문서와 결과

- [인지 DDI 150-scene 검증 보고서](results/val150/인지_DDI_150scene_검증_보고서.md)
- [인지 DDI v1 사전등록 절차](docs/perception_ddi_v1_preregistered_protocol.md)
- [공식 모델 등록표](docs/val150_model_registry.csv)
- [인지 DDI 1차 분석 보고서](docs/인지_DDI_1차_분석_보고서.md)
- [인지 연구 설계](docs/perception_research_plan.md)
- [DDI 연구 목적 및 타당성 검증 프레임워크](docs/research_objective_and_validation_framework.md)
- [모델 실행 계획](docs/perception_model_execution_plan.md)
- [임계값 근거표](docs/perception_threshold_evidence.csv)
- [Trainval01 keyframe shard 검증 보고서](docs/trainval01_keyframe_shard_report.md)
- [Mini 임계값 감사](results/v1.0-mini/audit/perception_threshold_audit.md)
- [Trainval01 shard 결과](results/trainval01-shard)
- [Mini 핵심 그래프](results/v1.0-mini/figures)
- [Google Drive 연구 폴더](https://drive.google.com/drive/folders/1q33citSr3VgEAPpUpRwyDbY3ck6L6LA4)

## 참고 기준

- [NuScenes 공식 tutorial/schema](https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html)
- [NuScenes object detection protocol](https://www.nuscenes.org/object-detection)
- [NuScenes paper](https://arxiv.org/abs/1903.11027)
- [KITTI object detection difficulty](https://www.cvlibs.net/datasets/kitti/eval_object.php)
- [MMDetection3D NuScenes guide](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/datasets/nuscenes.md)

## 데이터 정책

NuScenes 원본, 비공식 trainval ZIP, 모델 checkpoint, 전체 객체 단위 CSV 및 임시 prediction은 Git에 포함하지 않습니다. 이 저장소에는 재현 코드, 연구 문서와 검토 가능한 소형 요약 결과만 저장합니다.
