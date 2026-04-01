# Denoiser Module

## 1. 프로젝트 개요
본 모듈은 번호판 이미지에 대해 다양한 디노이징 기법을 적용하고,  
각 방법의 성능을 비교 및 분석하는 것을 목표로 한다.

---

## 2. 데이터 생성 방식
- 입력: YOLO로 분리된 번호판 이미지
- 처리: 다양한 노이즈를 인위적으로 삽입
- 출력: clean / noisy 이미지 쌍으로 구성된 데이터셋

---

## 3. 비교 대상 모델
- Classical (OpenCV 기반)
- DnCNN (CNN 기반 딥러닝 모델)
- Transformer 기반 디노이저

---

## 4. 평가 지표
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- 기타 필요 시 추가

---

## 5. 결과 비교
| Model        | PSNR | SSIM |
|-------------|------|------|
| Classical   |      |      |
| DnCNN       |      |      |
| Transformer |      |      |

각 모델의 디노이징 결과 샘플 사진 첨부 필요

---

## 6. 결론
- 각 모델의 성능 요약
- 최종적으로 가장 효과적인 방법에 대한 분석