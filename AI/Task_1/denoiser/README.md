# Denoiser Module

## 1. 프로젝트 개요
본 모듈은 번호판 이미지에 대해 다양한 디노이징 기법을 적용하고,  
각 방법의 성능을 비교 및 분석하는 것을 목표로 한다.

---

## 2. 폴더 및 구조 (Directory Structure)
본 디렉토리는 각 디노이징 모델의 성격과 역할을 명확히 나누어 관리합니다.

- **`classical/`**: OpenCV 등을 활용한 전통적인 이미지 필터링(Gaussian, Median 등) 방식의 코드
- **`dncnn/`**: CNN 기반의 고전적 딥러닝 디노이징 모델 코드 및 가중치
- **`transformer/`**: Transformer 아키텍처를 응용한 디노이징 실험 코드
- **`nafnet/`**: (최종 채택) 색상 보존 능력과 연산 효율이 극대화된 NAFNet 기반의 컬러 디노이징 아키텍처, 학습 및 평가 코드
- **`dataset/`**: 원본 이미지에 인위적인 노이즈를 씌워 Clean/Noisy 이미지 쌍(Pair)을 생성하고 관리하는 데이터 로더 스크립트
- **`utils/`**: 공통적으로 사용되는 평가 보조 함수나 유틸리티 모음

---

## 3. 데이터 생성 방식
- 입력: YOLO로 분리된 번호판 이미지
- 처리: 다양한 노이즈를 인위적으로 삽입
- 출력: clean / noisy 이미지 쌍으로 구성된 데이터셋

---

## 4. 비교 대상 모델
- Classical (OpenCV 기반)
- DnCNN (CNN 기반 딥러닝 모델)
- Transformer 기반 디노이저
- **NAFNet (Color Denoising 특화)**

---

## 5. 평가 지표
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- 기타 필요 시 추가

---

## 6. 결과 비교
| Model        | PSNR | SSIM |
|-------------|------|------|
| Classical   |      |      |
| DnCNN       | 23.26| 0.53 |
| Transformer |      |      |
| **NAFNet**  | **25.49**| **0.66** |

각 모델의 디노이징 결과 샘플 사진 첨부 필요

---

## 7. 결론
- 각 모델의 성능 요약
- 최종적으로 가장 효과적인 방법에 대한 분석