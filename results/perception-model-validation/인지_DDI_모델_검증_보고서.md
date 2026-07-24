# nuScenes 인지 DDI 모델 검증 보고서

## 1. 검증 질문

제안한 인지 DDI가 높을수록 서로 다른 자율주행 인지 모델의 실제 검출 오류가 일관되게 증가하는지 검증한다.

## 2. 데이터와 분할

- 완전 keyframe: 3,376개, 85개 장면
- 기준 개발용 공식 train: 2,462개 keyframe
- 최종 모델 검증용 공식 validation: 914개 keyframe, 23개 장면
- validation 카메라 추론: keyframe당 6개 카메라
- intermediate sweep는 사용하지 않았다.

## 3. 사전 동결한 난이도 경계

- train 분포 3분위 경계: Easy <= 59.29, Moderate <= 67.23, Hard > 67.23
- 이 경계는 validation 모델 오류를 보기 전에 train DDI 분포만으로 정했다.
- 발표안의 고정 33.33/66.67 경계도 비교 기준으로 유지했다.

## 4. 모델과 오류 정의

- 모델: FCOS3D, PGD의 공식 nuScenes monocular fine-tuned checkpoint
- 주 분석: confidence 0.05, 동일 클래스 중심거리 2m 이내를 true positive로 매칭
- 지표: false-negative rate, false positives, matched-box translation error
- 민감도 분석: 중심거리 1m/4m, confidence 0.10/0.20

## 5. 주 결과

- FCOS3D: Easy 0.228, Moderate 0.195, Hard 0.164; 엄격한 단조 증가=미충족
- PGD: Easy 0.292, Moderate 0.289, Hard 0.241; 엄격한 단조 증가=미충족
- FCOS3D 조정모형: DDI 10점 증가 OR=0.857 (95% CI 0.724-1.016, p=0.07533)
- PGD 조정모형: DDI 10점 증가 OR=0.842 (95% CI 0.683-1.038, p=0.108)
- FCOS3D 연속 DDI-frame FN Spearman rho=-0.052 (p=0.1199)
- PGD 연속 DDI-frame FN Spearman rho=-0.030 (p=0.3661)
- 두 모델 frame false-negative rate의 Spearman rho: 0.657
- 사전 정의된 수용 기준에 대한 현재 판정: **미지지**
- 민감도 조건 중 엄격한 Easy < Moderate < Hard 충족: 0/10개
- 고정 33.33/66.67 경계의 validation keyframe 수: Easy 34, Moderate 480, Hard 400

### 단순 기준선 비교

- FCOS3D / DDI: object FN AUC=0.487
- FCOS3D / raw_object_count: object FN AUC=0.466
- FCOS3D / object_distance: object FN AUC=0.704
- FCOS3D / inverse_lidar_points: object FN AUC=0.625
- PGD / DDI: object FN AUC=0.492
- PGD / raw_object_count: object FN AUC=0.473
- PGD / object_distance: object FN AUC=0.714
- PGD / inverse_lidar_points: object FN AUC=0.658

### 지표별 해석

주 분석에서 각 구성요소와 frame FN rate의 Spearman rho는 다음과 같다.

- perception_difficulty_score: FCOS3D -0.052, PGD -0.030
- count_score: FCOS3D 0.009, PGD 0.011
- visibility_mean_score: FCOS3D 0.178, PGD 0.296
- visibility_max_score: FCOS3D 0.092, PGD 0.104
- lidar_sparsity_mean_score: FCOS3D 0.238, PGD 0.281
- lidar_sparsity_max_score: FCOS3D 0.211, PGD 0.219
- truncation_mean_score: FCOS3D -0.123, PGD -0.138
- truncation_max_score: FCOS3D -0.154, PGD -0.170

평균 가시성 난이도는 두 모델에서 일관된 양의 관계를 보였다. 반면 객체 수는 거의 무관했고, truncation proxy는 두 모델 모두 역방향이었다. LiDAR 희소성 평균은 양의 관계였지만 카메라 모델 결과이므로 거리 교란을 포함한 간접 신호로만 해석해야 한다. 현재 합성 점수는 이 상반된 구성요소와 포화된 maximum 항을 평균하여 유효 신호를 상쇄했다.

## 6. 해석 원칙

- 두 모델에서 Easy < Moderate < Hard가 반복되고 연속 DDI 효과도 양수이면 단계 분류의 예측 타당성을 지지한다.
- 한 모델에서만 성립하거나 민감도 조건에 따라 방향이 바뀌면 모델 일반화 또는 강건성이 부족하다.
- 단계 경계는 실패하지만 연속 DDI와 오류가 관련되면 연속 지표만 유지하는 결론이 가능하다.
- DDI가 단순 객체 수·거리 기준선보다 낫지 않으면 현재 합성 방식의 추가 타당성은 입증되지 않는다.

## 7. 한계

- 현재 자료는 trainval01 shard의 85개 완전 장면이며 전체 nuScenes trainval 850개 장면이 아니다.
- 검증 모델 둘 다 monocular 계열이므로 센서 양식이 다른 LiDAR·fusion 모델로 외적 검증이 필요하다.
- 따라서 이번 결과는 camera-aligned 지표와 presentation_legacy_v0 기준선의 검증이며, LiDAR 희소성을 포함한 통합 인지 DDI의 완전 검증은 아니다.
- keyframe-only 실험이므로 temporal sweep 활용 난이도는 검증하지 않는다.
- train 3분위 경계는 객관적 정답이 아니라 오류를 보지 않고 동결한 잠정 운영 경계다.

## 8. 산출물

세부 수치와 재현 가능한 결과는 같은 폴더의 CSV 및 figures 디렉터리에 저장했다.
