# NuScenes 인지 DDI 1차 분석 보고서

## 1. 연구 목표

우리가 제안한 인지 난이도 지표가 높은 장면에서 실제 객체 검출 모델의 오류율이 증가하는지 검증한다. DDI 점수 자체를 정답으로 가정하지 않고, 여러 인지 모델의 오류를 외부 준거로 사용한다.

1차 연구의 범위는 NuScenes의 2Hz annotation keyframe이다. Intermediate sweep을 사용하는 temporal 모델은 별도 확장 실험으로 둔다.

## 2. 현재 데이터 및 분석 상태

- 공식 `v1.0-mini`: 10 scenes, 404 keyframes, 18,538 object annotations
- 검증 센서: 6 cameras + `LIDAR_TOP`
- 검증 결과: keyframe 파일, metadata, annotation, map 연결 모두 정상
- 산출 변수: 객체 수, VRU/차량 수, visibility, LiDAR/Radar point 수, ego 거리, 3D box 크기, camera truncation
- 분석 단위: scene, keyframe, object의 세 단계

Mini는 표본이 작고 어려운 장면 위주로 구성되어 있으므로 최종 임계값 선정에는 사용하지 않는다. 여기서는 코드 검증과 기준의 포화 여부를 진단하는 용도로만 사용한다.

## 3. 발표 기준의 1차 진단 결과

| 항목 | Mini 결과 | 해석 |
|---|---:|---|
| 객체 수 점수 10점 비율 | 92.1% | 현재 `C > 15` 또는 raw count 40 기준은 NuScenes 장면을 거의 구분하지 못함 |
| 최대 visibility 점수 10점 비율 | 94.6% | 장면에 심하게 가려진 객체가 하나만 있어도 거의 모든 장면이 최고점이 됨 |
| 최대 LiDAR 희소성 점수 10점 비율 | 100.0% | 최악 객체 하나를 사용하는 집계 방식은 완전히 포화됨 |
| LiDAR 0~4점 객체 비율 | 53.5% | `5/20/50점` 구간은 객체 수준에서도 낮은 구간에 과도하게 집중될 가능성이 있음 |
| 공식 평가에서 제외되는 0 sensor-point 객체 | 18.7% | 공식 평가는 LiDAR와 Radar 점의 합이 0인 객체를 제거하므로 평가 대상과 DDI 대상의 정렬이 필요함 |
| 발표안 truncation 0.05 이하 | 95.3% | 360도 카메라에서 truncation은 대부분 낮아 현재 점수 기여도가 작음 |
| 고정 0~33/33~67/67~100 분류의 Low 비율 | 0.0% | 현재 합산점수는 Low/Medium/High 분류 기준과 맞지 않음 |

현재 결과는 객체 수, visibility, LiDAR point 수가 쓸모없다는 뜻이 아니다. **변수는 타당한 후보지만 임계값과 장면 집계 방식이 NuScenes 분포에 맞지 않는다**는 의미다.

현재 합성점수는 객체 수 항목 1개와 visibility, LiDAR 희소성, truncation 항목을 각각 평균·최대값 2개씩 사용한다. 따라서 별도의 근거 없이 영역 가중치가 `1:2:2:2`가 되는 문제도 있다. 기존 점수는 `presentation_legacy_v0` 기준선으로만 보존하고, 수정 점수는 각 영역을 한 번씩 집계하거나 Train split의 모델 오류로 가중치를 추정해야 한다.

## 4. 근거자료와 발표 기준의 차이

### Visibility

NuScenes는 객체가 6개 카메라에서 보이는 비율을 `0~40%`, `40~60%`, `60~80%`, `80~100%`의 네 구간으로 공식 제공한다. 따라서 visibility 구간 자체는 객관적인 출처가 있다.

다만 공식 구간은 annotation 속성이지 장면 난이도 점수가 아니다. 장면 점수에서는 최대값 대신 다음 비율을 사용한다.

- visibility token 1 객체 비율
- token 1~2 객체 비율
- 객체별 visibility severity 평균
- 클래스와 거리를 통제한 visibility별 오류율

### Truncation

KITTI의 공식 난이도 기준은 Easy/Moderate/Hard에 대해 최대 truncation `0.15/0.30/0.50`을 사용한다. 발표안의 첫 경계 `0.05`는 KITTI 공식 Easy 기준과 다르다.

NuScenes는 공식 truncation 난이도 라벨을 제공하지 않는다. 현재 값은 3D box를 카메라에 투영해 계산한 파생값이므로, `0.05/0.30/0.50`과 `0.15/0.30/0.50`을 후보로 비교하되 최종 선택은 모델 오류와의 관계로 결정한다.

### LiDAR point 수

`num_lidar_pts`는 NuScenes 공식 annotation 필드이지만 `5/20/50점`은 공식 난이도 기준이 아니다. LiDAR point 수는 객체 종류, 거리, 크기에 크게 좌우되므로 같은 10점짜리 자동차와 보행자를 직접 비교하면 편향될 수 있다.

Mini 교차분석에서 차량의 LiDAR point 중앙값은 거리 0~20m에서 143점, 20~40m에서 7점, 40~60m에서 1점, 60m 이상에서 0점이었다. VRU는 같은 거리 구간에서 각각 14점, 3점, 1점, 1점이었다. 따라서 고정 point 기준은 독립적인 난이도라기보다 거리와 클래스 효과를 함께 측정한다. 모델 오류와의 관계를 검정할 때 거리·클래스를 통제하고, 상대적 희소성을 별도로 계산해야 한다.

최종 분석에서는 다음 두 방식을 함께 비교한다.

1. 발표안의 고정 구간 `0~4/5~19/20~49/50+`
2. 클래스 및 거리 구간별 정규화된 `log(1 + num_lidar_pts)` 또는 분위수

## 5. 수정된 인지 난이도 구조

모든 모델에 하나의 점수를 강제로 적용하지 않고 공통 장면 특성과 센서별 난이도를 분리한다.

### 공통 장면 특성

- 객체 수와 밀집도
- VRU 비율
- 클래스 구성
- 객체 거리 분포
- 객체 크기 분포

### Camera 난이도 `P_camera`

- visibility token 분포
- 2D projected size
- truncation
- 조도 및 날씨
- 카메라별 객체 분포

### LiDAR 난이도 `P_lidar`

- 객체별 LiDAR point 수
- 클래스·거리 대비 point 희소성
- zero-LiDAR-point 객체 비율
- 원거리 객체 비율
- 작은 객체 및 VRU 비율

### Fusion 난이도 `P_fusion`

- Camera와 LiDAR 난이도의 결합
- 한 센서가 취약할 때 다른 센서가 보완하는 정도
- 센서별 오류의 일치 및 불일치

## 6. 모델 오류 검증 방법

객체가 많은 장면은 단순 오류 개수도 자연스럽게 증가하므로 raw error count를 주 지표로 사용하지 않는다.

- False Negative: 전체 정답 객체 중 놓친 객체의 비율
- False Positive: keyframe당 잘못 검출한 객체 수
- Localization error: 매칭된 객체의 중심 위치 오차
- Class-specific recall: 자동차, 보행자, 자전거 등 클래스별 재현율
- Distance-specific recall: 거리 구간별 재현율
- 공식 NuScenes 지표: mAP, mATE, mASE, mAOE, mAVE, mAAE, NDS

객체 수준 오류 모델에서는 클래스, 거리, 객체 크기, 도시, 장면을 통제한다. 동일 scene의 keyframe이 반복 측정이라는 점을 고려해 scene 단위 bootstrap 신뢰구간을 사용한다.

## 7. Trainval 도착 후 실행 순서

1. 850 scenes와 34,149 keyframes가 모두 있는지 검사한다.
2. 6개 카메라와 LiDAR 파일 누락 및 annotation 연결을 검사한다.
3. trainval 전체의 객체·keyframe·scene 특징을 추출한다.
4. Train split에서만 분포와 후보 임계값을 계산한다.
5. 고정 기준, KITTI 후보 기준, 경험적 분위수 기준을 비교한다.
6. 단일 keyframe Camera 모델과 LiDAR 모델의 추론 결과를 생성한다.
7. 예측과 정답을 객체 단위로 매칭한다.
8. DDI Low/Medium/High별 오류율과 신뢰구간을 계산한다.
9. 단순 기준인 객체 수, 거리, visibility 단독 모델과 비교한다.
10. Validation split에서 최종 기준을 한 번만 평가한다.

## 8. 현재 결론

인지 DDI의 기본 변수 선택은 연구 가치가 있지만 발표 당시의 점수화 방식은 그대로 사용하기 어렵다. 특히 최대값 집계는 거의 모든 장면을 최고 난이도로 만들기 때문에 제거하거나 비율·분위수 집계로 변경해야 한다.

최종적으로 입증해야 할 명제는 "이 수치가 어려워 보인다"가 아니라 다음 문장이다.

> Train split에서 정한 인지 난이도 기준이 Validation split의 서로 다른 인지 모델에서도 객체당 오류 확률 증가를 안정적으로 예측한다.

## 9. 주요 출처

- [KITTI Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php)
- [NuScenes 공식 튜토리얼 및 schema](https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html)
- [NuScenes Object Detection 평가 방식](https://www.nuscenes.org/object-detection)
- [NuScenes 논문](https://arxiv.org/abs/1903.11027)
