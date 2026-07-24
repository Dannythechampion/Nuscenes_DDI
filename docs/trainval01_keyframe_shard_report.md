# NuScenes Trainval01 Keyframe Shard 검증 보고서

## 1. 결론

Google Drive에 저장한 `nuScenes-v1.0-trainval-keyframes-kaggle.zip`은 전체 keyframe-only trainval이 아니다. 압축 내부 marker와 파일 구조를 확인한 결과 다음 공식 archive 두 개를 합친 package다.

- `v1.0-trainval_meta.tgz`: 전체 850 scenes의 metadata
- `v1.0-trainval01_blobs.tgz`: trainval sensor blob 중 첫 번째 shard

따라서 metadata에는 34,149개 sample이 있지만 실제 keyframe sensor 파일은 3,376개 sample만 존재한다.

## 2. 무결성 및 Coverage

| 항목 | 결과 |
|---|---:|
| Metadata scenes | 850 |
| Metadata samples | 34,149 |
| Metadata annotations | 1,166,187 |
| 완전 keyframes | 3,376 |
| 누락 keyframes | 30,773 |
| Keyframe coverage | 9.886% |
| 완전 scenes | 85 |
| 부분 scenes | 0 |
| 공식 train split scenes | 62 |
| 공식 validation split scenes | 23 |

완전 keyframe은 6개 camera와 `LIDAR_TOP` 파일이 모두 존재하는 sample로 정의했다. 85개 scene은 scene 내부 sample이 모두 존재하며, 파일 일부만 남은 partial scene은 없다.

## 3. Package 구성

| 디렉터리 | 파일 수 | 크기 |
|---|---:|---:|
| `samples` | 40,512 | 5.18 GiB |
| `sweeps` | 217,595 | 33.23 GiB |
| `maps` | 13 | 0.45 GiB |
| metadata | 13 | 2.42 GiB |

`samples`에는 6 camera, 5 radar, 1 LiDAR channel이 각각 3,376개 들어 있다. 본 연구는 keyframe 중심이므로 추출 후 `sweeps`와 로컬 ZIP을 제거했으며, 원본 ZIP은 Google Drive에 유지했다.

## 4. 인지 특징 분석

완전 keyframe만 선택해 3,376개 sample, 107,206개 object annotation, 85개 scene의 특징을 추출했다.

| 진단 항목 | 결과 |
|---|---:|
| 객체 LiDAR point `0-4` | 53.82% |
| zero LiDAR+Radar point object | 15.13% |
| visibility token 4 | 51.24% |
| visibility token 1 | 28.11% |
| truncation `<=0.05` | 94.67% |
| count score 최고 bin | 78.11% |
| visibility max 최고 bin | 90.97% |
| LiDAR sparsity max 최고 bin | 97.10% |

기존 고정 DDI 구간은 Low 3.64%, Medium 59.09%, High 37.26%로 분류됐다. Mini에서 Low가 0%였던 문제는 줄었지만 여전히 component max 포화가 크다.

## 5. 거리·클래스 조건부 LiDAR 분포

| 그룹 | 0-20m median | 20-40m median | 40-60m median | 60m+ median |
|---|---:|---:|---:|---:|
| Vehicle | 138 | 9 | 2 | 1 |
| VRU | 13 | 3 | 1 | 1 |
| Static | 12 | 2 | 1 | 0 |

동일한 LiDAR point 임계값을 모든 객체에 적용하면 거리와 클래스 효과를 난이도로 잘못 해석한다. `P_lidar`는 클래스·거리 구간 내 percentile 또는 기대 point 대비 잔차로 정의해야 한다.

## 6. 연구에 사용할 수 있는 범위

이 shard로 가능한 작업:

1. 특징 추출 및 DDI 계산 pipeline 검증
2. 23개 validation scene에서 Camera/LiDAR 모델 pilot 추론
3. DDI 구간별 FN/FP 변화의 예비 분석
4. 임계값과 집계 방식의 sensitivity analysis

이 shard만으로 주장하면 안 되는 내용:

1. 전체 NuScenes trainval을 대표하는 최종 분포
2. 전체 validation split의 공식 mAP/NDS
3. 환경·지역·날씨 전반에 일반화된 최종 DDI 임계값
4. 이 shard에서 학습한 모델의 full trainval 성능

## 7. 다음 단계

1. 23개 validation scene에 Camera baseline prediction을 생성한다.
2. `match_nuscenes_detection_errors.py`로 객체별 FN/FP와 localization error를 계산한다.
3. `P_camera`와 오류율의 관계를 먼저 검증한다.
4. 전체 연구 결론 전에는 나머지 공식 blob에서 `samples`만 순차 추출·병합한다.
5. 병합 후 `validate_nuscenes_keyframes.py`가 34,149개 sample을 모두 통과하는지 확인한다.

공식 nuScenes devkit은 devkit 사용에 metadata와 samples가 필요하고 sweeps는 선택 사항이라고 설명한다. 전체 trainval을 구성할 때는 모든 archive를 받아 동일 폴더에 병합해야 한다.

- [nuScenes devkit dataset structure](https://github.com/nutonomy/nuscenes-devkit#dataset-download)
- [검증 결과 JSON](../results/trainval01-shard/keyframe_validation.json)
- [Shard 감사 결과](../results/trainval01-shard/audit/perception_threshold_audit.md)
