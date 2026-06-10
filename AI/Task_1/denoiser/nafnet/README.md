# NAFNet Denoising Module

본 디렉토리는 기존 DnCNN 모델을 대체하는 **NAFNet** 기반의 새로운 디노이징 아키텍처와 학습 파이프라인을 포함하고 있습니다.

## 핵심 변경 사항
- **모델 구조**: `model.py`에 NAFNet(SimpleGate, SCA 적용)을 새롭게 구현하여 기존 모델 대비 파라미터 수를 절반(약 300K)으로 줄이면서도 연산 효율을 극대화했습니다.
- **손실 함수 (Loss)**: 단순 픽셀 오차(MSE) 대신 **Charbonnier Loss**와 **SSIM Loss**를 도입하여 글자의 윤곽선(Edge)과 구조적 선명도를 최우선으로 보존하도록 설계되었습니다.

## 학습 방법 (Training)

훈련은 별도의 인위적인 데이터 비율 조작 없이 자연스러운 환경에서 수행되었습니다.
기존 DnCNN은 원본 보존을 위해 깨끗한 이미지를 강제로 40% 섞어 주입(`--identity_ratio 0.4`)했으나, NAFNet은 아키텍처 자체가 원본 보존(Identity Mapping)에 탁월하여 해당 옵션을 기본값(0%)으로 두고 혹독하게 노이즈 제거에만 집중하여 학습시켰습니다.

### 1. 기본 학습 실행
```bash
# NAFNet 모델 학습 (기본 10 Epoch, SSIM Loss 적용)
python train.py --epochs 10 --use_ssim_loss --data_dir ../data/dataset_color
```

### 2. 트레이드오프 검증 실험 (선택)
과거 방식처럼 깨끗한 데이터를 강제 주입했을 때의 부작용(노이즈 제거 능력 하락)을 검증하려면 아래 명령어를 사용합니다.
```bash
# 40% 강제 주입 테스트
python train.py --epochs 10 --use_ssim_loss --identity_ratio 0.4 --data_dir ../data/dataset_color
```

## 성능 평가 (Evaluation)
학습된 모델의 검증 및 벤치마크는 아래 스크립트를 사용합니다.
```bash
python evaluate.py --data_dir ../data/dataset_color
```
- 평가 결과, 기존 모델 대비 **PSNR이 +2.23dB 상승**하였고, **SSIM이 0.53에서 0.66으로 대폭 상승**하였습니다.

## 파일 구성 및 쓰임새 (File Structure)
본 폴더 내에 포함된 각 스크립트의 역할은 다음과 같습니다.

- **`model.py`**: NAFNet의 핵심 신경망 아키텍처(SimpleGate, SCA 블록 포함)를 정의하는 모델 클래스 파일입니다.
- **`train.py`**: 데이터로더 설정, 손실 함수(Charbonnier, SSIM), 그리고 학습 루프를 제어하는 메인 훈련 스크립트입니다. `--identity_ratio` 등의 실험적 하이퍼파라미터를 여기서 조작할 수 있습니다.
- **`evaluate.py`**: 훈련이 끝난 가중치 모델을 불러와, 테스트 데이터셋을 대상으로 전체 PSNR 및 SSIM 점수를 일괄 측정하고 잔차(Residual) 분석을 돕는 평가 스크립트입니다.
- **`generate_samples.py`**: 노이즈 이미지(Input)가 모델을 통과하며 제거된 노이즈(Residual)와 최종 결과물(Output)로 변환되는 과정을 시각화하여 모자이크 이미지로 저장합니다.
- **`generate_dncnn_samples.py`**: 구형 모델(DnCNN)의 디노이징 결과를 시각화하여, NAFNet과의 성능(특히 컬러 보존 능력) 비교 증빙 자료를 생성하는 데 쓰입니다.
- **`generate_zoom.py`**: 번호판 글씨의 아주 미세한 윤곽선이 뭉개지지 않고 보존되었음을 증명하기 위해, 픽셀 단위로 확대한(Zoom-in) 시각화 자료를 뽑아내는 스크립트입니다.
