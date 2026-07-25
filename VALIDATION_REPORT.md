# 인지 DDI 공식 validation 검증

현재 기준 보고서는 [nuScenes 인지 DDI v1 150-scene 검증 보고서](results/val150/인지_DDI_150scene_검증_보고서.md)다.

## 범위

- 공식 nuScenes validation: 150 scenes, 6,019 keyframes
- Camera models: FCOS3D, PGD
- LiDAR models: PointPillars, CenterPoint
- Camera 파일 36,114개, LiDAR 10-frame 요구 파일 57,916개, 누락 0개
- 모델별 예측 token 6,019개, 누락·예상 밖 token 0개

## 결과

네 모델 모두 Easy, Moderate, Hard 순서로 원시 frame FN rate가 증가했고 20개 민감도 조건에서도 방향이 유지됐다. 하지만 조정된 연속 DDI 효과가 모든 모델에서 유의한 양의 효과가 아니었고, 객체 단위 효과는 역방향이었다.

따라서 현재 DDI v1 합성 지표의 사전등록 판정은 **지지 실패**다. 가림과 LiDAR 5-point 희소성은 후보 component로 유지하지만 밀집도 가중치, 동일 가중 합성, frame 집계는 재설계가 필요하다.

이전에 기록한 trainval01의 23-scene·914-keyframe 결과는 pilot이다. 본 문서와 `results/val150` 결과가 공식 전체 validation 본실험을 대체한다.
