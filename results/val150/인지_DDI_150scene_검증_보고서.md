# nuScenes 인지 DDI v1 150-scene 검증 보고서

## 1. 결론 요약

이번 실험은 KITTI, Waymo, nuPlan 등에서 가져온 인지 난이도 기준을 nuScenes 공식 validation 150개 scene, 6,019개 keyframe에 적용하고, nuScenes로 학습된 공식 모델 네 개의 실제 검출 오류와 비교했다.

최종 결론은 **현재 인지 DDI v1 합성 지표는 사전등록한 전체 타당성 기준을 충족하지 못했다**이다.

- Camera와 LiDAR 모두 두 모델에서 `Easy < Moderate < Hard` 순서로 frame FN rate가 증가했다.
- 총 20개 매칭 민감도 조건도 모두 같은 단조 증가를 보였다.
- 그러나 거리, 클래스, 객체 수, box 크기, 장소, scene 군집을 통제한 연속 DDI 효과는 모델 전체에서 유의한 양의 효과가 아니었다.
- 객체 단위 조정 효과는 오히려 역방향이었다.
- Camera 합성 DDI는 가장 강한 단일 성분인 가림보다 약했다.
- LiDAR 합성 DDI는 frame 상관에서는 단일 성분보다 조금 강했지만, 객체 FN AUC는 원시 LiDAR point 수 기준보다 낮았다.

따라서 외부 기준이 전부 무의미하다는 결론은 아니다. **외부 기준으로 만든 ordinal 난이도 그룹은 오류를 구분했지만, 현재의 연속 점수화·동일 가중치·frame 집계 방식이 독립적이고 일반화 가능한 DDI라고 입증되지는 않았다.**

## 2. 연구 질문과 원칙

연구 질문은 다음과 같다.

> 외부 데이터셋과 선행 기준에서 옮긴 인지 난이도 지표가 nuScenes에서 서로 다른 두 개 이상의 모델의 실제 오류 증가를 반복적으로 예측하는가?

KITTI, Waymo, H3D, nuPlan은 실험 데이터로 섞지 않았다. 이 자료들은 임계값과 개념의 근거로만 사용했고, 모델 추론과 오류 검증은 모두 nuScenes에서 수행했다. validation 오류를 본 뒤 임계값을 분위수에 맞춰 조정하지 않았다.

## 3. 데이터 완전성

- Dataset: nuScenes `v1.0-trainval`, 공식 `val`
- Scene: 150개
- Keyframe: 6,019개
- Camera keyframe 파일: 36,114개
- LiDAR keyframe 파일: 6,019개
- 현재 frame과 과거 sweep을 포함한 고유 LiDAR 요구 파일: 57,916개
- 누락 파일: 0개
- Detection 평가 대상 GT 객체: 121,871개
- 평가 대상 객체가 없는 keyframe: 71개

PointPillars는 현재 frame과 과거 10개 sweep, CenterPoint는 현재 frame과 과거 9개 sweep을 공식 config에 따라 사용했다.

## 4. 사전등록 지표

### Camera

- 가시성: nuScenes visibility `80-100%`는 Easy, `40-80%`는 Moderate, `0-40%`는 Hard
- 잘림: KITTI의 15%, 30% 경계를 nuScenes 3D box의 best-camera 투영 잘림 비율에 적용
- 객체 난이도: 가시성과 잘림 중 더 어려운 등급
- Frame 난이도: 평가 대상 객체 ordinal의 75 percentile을 올림

### LiDAR

- Waymo의 5-point 경계를 이전
- `>5 points`는 Easy, `1-5`는 Moderate, `0`은 Hard
- Frame 난이도: 평가 대상 객체 ordinal의 75 percentile을 올림

### 밀집도

- nuPlan의 `8m 내 moving vehicle > 6` 및 ego speed `>6m/s` 조건을 번역
- 발표안의 `2*VRU + vehicle + 0.5*static`과 40-object 경계는 출처가 확정된 기준이 아니므로 탐색 가설로 분리

### 분포

| Modality | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Camera | 870 | 2,680 | 2,469 |
| LiDAR | 1,467 | 4,477 | 75 |

LiDAR Hard가 75개 keyframe, 26개 scene에만 존재하므로 Hard 추정치는 다른 그룹보다 불확실하다.

## 5. 모델 재현성

| Modality | Model | 측정 mAP | 측정 NDS | 공식 mAP | 공식 NDS |
|---|---|---:|---:|---:|---:|
| Camera | FCOS3D | 32.12 | 39.48 | 32.10 | 39.30 |
| Camera | PGD | 34.58 | 41.11 | 34.60 | 41.10 |
| LiDAR | PointPillars | 33.69 | 48.55 | 34.33 | 49.10 |
| LiDAR | CenterPoint | 55.49 | 64.09 | 56.11 | 64.61 |

네 모델 모두 6,019개 token을 정확히 예측했고 누락·예상 밖 token은 0개였다. PointPillars의 최초 후보였던 MMDetection3D v0.7.0 checkpoint는 현재 1.x 좌표 규약과 호환되지 않아 제외했으며, 현재 1.x model zoo의 공식 checkpoint로 다시 실행했다.

## 6. 주 결과

주 분석은 동일 클래스 중심거리 2m, confidence 0.05 매칭이다.

| Model | Easy FN | Moderate FN | Hard FN | 단조 증가 |
|---|---:|---:|---:|---:|
| FCOS3D | 0.2138 | 0.2158 | 0.2574 | Yes |
| PGD | 0.2543 | 0.2895 | 0.3480 | Yes |
| PointPillars | 0.0838 | 0.1445 | 0.2821 | Yes |
| CenterPoint | 0.0555 | 0.0806 | 0.2508 | Yes |

원시 그룹 비교만 보면 ordinal 지표는 네 모델 모두에서 기대 방향을 만족한다. 1m/2m/4m 거리와 0.05/0.10/0.20 confidence를 바꾼 Camera 10개, LiDAR 10개 조건도 모두 단조 증가했다.

## 7. 조정 분석과 추가 타당성

Scene-clustered logistic regression의 frame DDI 결과는 다음과 같다.

| Model | OR/point | 95% CI | p-value |
|---|---:|---:|---:|
| FCOS3D | 1.0018 | 0.9070-1.1065 | 0.9711 |
| PGD | 1.0009 | 0.9178-1.0915 | 0.9839 |
| PointPillars | 1.0803 | 0.9667-1.2072 | 0.1733 |
| CenterPoint | 1.0047 | 0.8884-1.1363 | 0.9399 |

네 모델 모두 신뢰구간이 1을 포함한다. 객체 단위 DDI의 OR은 Camera 0.697/0.602, LiDAR 0.767/0.760으로 유의한 역방향이었다. 따라서 단순 그룹 차이를 DDI의 독립 효과로 해석할 수 없다.

Camera의 주 조건 Spearman rho는 FCOS3D 0.194, PGD 0.292였지만, 가림 단독은 각각 0.197, 0.317로 더 강했다. LiDAR 합성 DDI는 PointPillars 0.240, CenterPoint 0.168로 `<=5 point 비율` 단독의 0.233, 0.149보다 조금 강했다. 반면 객체 FN AUC에서 LiDAR 합성 점수는 0.669/0.647이고 inverse point 수는 0.718/0.685로 단일 원시 기준이 더 강했다.

## 8. 지표별 판정

### 객체 밀집도와 복잡도

- 번역한 nuPlan event는 6,019개 중 0개여서 nuScenes 분류 기준으로 변별력이 없었다.
- 발표안의 weighted complexity `>15`는 4,734개 keyframe을 Hard로 보내 포화됐다.
- VRU:vehicle:static의 회귀계수는 모델마다 방향과 유의성이 달랐고 `2:1:0.5` 순서를 지지하지 않았다.
- 판정: 현재 임계값과 가중치는 기각하고, 밀집도를 객체 수 하나가 아니라 거리·상호작용·ego 상태와 함께 재정의해야 한다.

### Camera 가림

- 오류와 일관된 양의 원시 관계가 나타났고 Camera 합성 점수보다 강한 경우가 있었다.
- 판정: 유망한 독립 component로 유지하되, 현재 동일 가중 합성 방식의 타당성 근거로 사용해서는 안 된다.

### Camera 잘림

- KITTI의 비율 경계는 해상도와 무관하다는 장점이 있지만, 이번 nuScenes 모델 오류에서 독립적인 강한 예측력을 보이지 못했다.
- 판정: 경계의 외적 근거는 유지할 수 있으나 nuScenes용 component로 채택하려면 별도 검증 또는 camera별 가시 영역 계산 개선이 필요하다.

### LiDAR 희소성

- Waymo 5-point 기준의 ordinal 그룹은 두 LiDAR 모델에서 강한 단조 오류 증가를 보였다.
- 그러나 Hard frame이 매우 적고, 객체 수준에서는 원시 point 수가 합성 DDI보다 더 잘 예측했다.
- 판정: 5-point 기준은 유지할 근거가 있으나 frame 집계와 density 결합 방식은 재설계해야 한다.

## 9. 최종 판정

| 조건 | Camera | LiDAR |
|---|---:|---:|
| 세 그룹 모두 존재 | Pass | Pass |
| 두 모델 주 분석 단조 증가 | Pass | Pass |
| 민감도 분석의 80% 이상 단조 | Pass | Pass |
| 모든 모델에서 조정 frame 효과 양수·유의 | Fail | Fail |
| 모든 모델에서 조정 object 효과 양수·유의 | Fail | Fail |
| 합성 점수가 최강 단일 성분보다 강함 | Fail | Pass |
| 전체 사전등록 지지 | **Fail** | **Fail** |

이 결과는 실패한 규칙을 validation 분포에 맞춰 고치지 않고 그대로 보존한 반증 결과다.

## 10. 다음 연구 절차

1. 외부 출처가 있는 가시성, truncation, 5-point 경계는 후보 component로 유지한다.
2. 밀집도 `2:1:0.5`, 40-object, 동일 가중 평균, frame P75 집계는 확정 기준에서 제외한다.
3. 공식 train split 또는 train 내부 교차검증에서 새 집계식과 가중치를 개발한다.
4. 모델·거리·클래스별 상호작용과 비선형 효과를 포함하되, 해석 가능한 component 수를 유지한다.
5. 이번 공식 val은 이미 확인했으므로 재설계 지표의 최종 검증에는 다른 held-out cohort 또는 외부 데이터셋의 공통 변수만 사용한다.
6. 판단·제어 DDI와 통합할 때도 인지 DDI v1을 확정 점수로 사용하지 말고, 검증된 component와 불확실성을 따로 전달한다.

## 11. 재현 산출물

- 사전등록 절차: `docs/perception_ddi_v1_preregistered_protocol.md`
- 모델·checkpoint 등록표: `docs/val150_model_registry.csv`
- 모델 성능: `results/val150/model_performance_summary.csv`
- 예측 token 감사: `results/val150/prediction_token_audit.csv`
- Camera 분석: `results/val150/camera-analysis/`
- LiDAR 분석: `results/val150/lidar-analysis/`
- 데이터 완전성: `results/val150/val150_materialization.json`

대용량 원시 특징표, 예측 JSON, 객체 매칭표는 Git에서 제외하고 Google Drive 연구 폴더에 보관한다.
