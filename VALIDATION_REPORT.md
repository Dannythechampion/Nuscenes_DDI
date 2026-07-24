# 인지 DDI Keyframe 모델 검증

현재 보유한 nuScenes trainval01 keyframe shard에서 연구 절차에 따라 FCOS3D와 PGD를 실행했다.

## 범위

- 완전 keyframe: 3,376개, 85개 장면
- 경계 개발: 공식 train 2,462개 keyframe, 62개 장면
- 최종 모델 검증: 공식 validation 914개 keyframe, 23개 장면
- 모델별 입력: 5,484개 카메라 관측
- 모델: FCOS3D, PGD
- 주 분석: confidence 0.05, 동일 클래스 중심거리 2m 매칭
- 민감도 분석: 중심거리 1m/4m, confidence 0.10/0.20

## 결론

현재 `presentation_legacy_v0` 합성 DDI는 사전 정의된 타당성 기준을 충족하지 못했다.

- FCOS3D frame FN rate: Easy 0.228, Moderate 0.195, Hard 0.164
- PGD frame FN rate: Easy 0.292, Moderate 0.289, Hard 0.241
- 두 모델 모두 기대한 `Easy < Moderate < Hard` 오류 증가가 나타나지 않았다.
- 10개 주·민감도 조건 중 엄격한 단조 증가를 충족한 조건은 0개였다.
- 평균 가시성 난이도는 두 모델에서 일관된 양의 관계를 보였다.
- 객체 수는 거의 무관했고, 현재 truncation proxy는 역방향이었다.
- 통합 점수의 객체 FN AUC는 FCOS3D 0.487, PGD 0.492로 거리 단독 기준선 0.704, 0.714보다 낮았다.

따라서 개별 인지 변수 전체가 무효라는 뜻은 아니다. 유효한 평균 가시성 신호와 포화된 maximum 항, 역방향 truncation 항을 같은 비중으로 평균한 현재 합성 방식이 지지되지 않았다는 뜻이다.

두 모델이 모두 monocular 계열이므로 이번 결과는 camera-aligned 지표의 모델 반복 검증이다. LiDAR 희소성을 포함한 통합 인지 DDI를 확정하려면 single-keyframe LiDAR 또는 fusion 모델을 추가해야 한다.

## 파일

- [전체 한국어 검증 보고서](results/perception-model-validation/인지_DDI_모델_검증_보고서.md)
- [재현 절차](docs/keyframe_perception_validation_protocol.md)
- [주 결론 CSV](results/perception-model-validation/primary_conclusion.csv)
- [지표별 타당성 CSV](results/perception-model-validation/component_validity.csv)
- [난이도 구간 결과 CSV](results/perception-model-validation/difficulty_group_results.csv)
- [검증 그림](results/perception-model-validation/figures/difficulty_vs_false_negative_rate.png)
- [Google Drive 원본·보고서 압축 파일](https://drive.google.com/drive/folders/14eq8PtT756JZLJW_scNZFIyXGFQL5dKs)

Drive 원본 ZIP SHA-256: `7D5B6188E984CDA2A4016C4D1A4B8F258A6D32678D48ED3B3B42CF4EAB16665A`

Drive 보고서 ZIP SHA-256: `46B6C9198DB3A6314556A0071AF3CC287AFE440A5362251C8044DD0A07E38C2D`
