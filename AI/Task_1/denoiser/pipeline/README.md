# Denoising Pipeline

## 1. 개요

본 모듈은 단일 이미지 입력을 받아 디노이징 모델을 통과시킨 뒤, 결과 이미지를 저장하는 공통 파이프라인이다.

현재는 DnCNN 기반 디노이저를 기본 모델로 사용한다. 다만 파이프라인 구조는 모델 교체가 가능하도록 작성되어 있으므로, 향후 Transformer 기반 디노이저 등 다른 모델로 확장할 수 있다.

---

## 2. 주요 기능

- 단일 이미지 입력 처리
- 이미지 전처리 수행
- PyTorch 기반 디노이징 모델 추론
- 결과 이미지 후처리 수행
- 결과 이미지 파일 저장
- 모델 구조 교체 가능
- state_dict 기반 checkpoint 로드 지원

---

## 3. 디렉토리 구조

```text
pipeline/
├── denoising_pipeline.py
├── input.png
├── output.png
└── README.md
```

각 파일의 역할은 다음과 같다.

- `denoising_pipeline.py`
  - 단일 이미지 디노이징 추론 파이프라인 코드

- `input.png`
  - 파이프라인에 입력으로 사용할 이미지

- `output.png`
  - 디노이징 결과 이미지
  - 실행 시 자동 생성되며, 기존 파일이 있을 경우 덮어쓰기된다.

- `README.md`
  - 파이프라인의 목적, 사용법, 모델 교체 방법을 설명하는 문서

---

## 4. 동작 흐름

파이프라인은 다음 순서로 동작한다.

```text
input.png
↓
preprocess
↓
denoising model inference
↓
postprocess
↓
output.png
```

세부 과정은 다음과 같다.

1. 입력 이미지 로드
2. Grayscale 변환
3. 모델 입력 크기로 resize
4. 0~1 범위로 정규화
5. Tensor 변환
6. 모델 추론 수행
7. 결과 Tensor를 이미지 형식으로 변환
8. 원본 해상도로 resize
9. output.png로 저장

---

## 5. 전처리 과정

입력 이미지는 모델에 전달되기 전에 다음 과정을 거친다.

- OpenCV를 이용해 Grayscale 이미지로 로드
- 모델 입력 크기인 `(48, 128)`로 resize
- 픽셀 값을 0~255 범위에서 0~1 범위로 정규화
- `float32` Tensor로 변환
- 모델 입력 형식인 `[1, 1, H, W]` 형태로 차원 확장

입력 이미지의 원본 해상도는 후처리 과정에서 다시 사용하기 위해 별도로 저장된다.

---

## 6. 후처리 과정

모델 출력은 다음 과정을 거쳐 이미지로 변환된다.

- 모델 출력 Tensor를 CPU로 이동
- NumPy 배열로 변환
- 0~1 값을 0~255 범위로 변환
- `uint8` 이미지로 변환
- 원본 이미지 해상도로 resize
- 지정된 경로에 이미지 파일로 저장

---

## 7. 실행 방법

`pipeline` 폴더 안에 `input.png`를 준비한 뒤 다음 명령어를 실행한다.

```bash
python denoising_pipeline.py
```

실행이 완료되면 같은 폴더에 `output.png`가 생성된다.

---

## 8. 설정 변경

설정은 `Config` 클래스에서 수정할 수 있다.

```python
class Config:
    input_path = "input.png"
    output_path = "output.png"

    image_size = (48, 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "../dncnn/best_model.pt"
```

주요 설정 항목은 다음과 같다.

| 설정 | 설명 |
|---|---|
| `input_path` | 입력 이미지 경로 |
| `output_path` | 결과 이미지 저장 경로 |
| `image_size` | 모델 입력 해상도 |
| `device` | 추론에 사용할 장치 |
| `checkpoint_path` | 학습된 모델 weight 경로 |

---

## 9. 모델 교체 방법

현재 파이프라인은 DnCNN 모델을 사용한다.

모델을 교체하려면 다음 두 부분을 수정하면 된다.

### 9.1 `build_model()` 함수 수정

`build_model()` 함수는 파이프라인에서 사용할 모델 객체를 생성하는 부분이다.

현재는 DnCNN 모델을 생성한다.

```python
def build_model():
    return DnCNN(in_channels=1, depth=17, features=64)
```

Transformer 기반 모델로 변경하려면 해당 함수에서 `TransformerDenoiser` 객체를 생성하도록 수정하면 된다.

예시:

```python
def build_model():
    return TransformerDenoiser(
        image_size=(48, 128),
        in_channels=1,
        embed_dim=64,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
    )
```

### 9.2 `checkpoint_path` 수정

모델을 변경하면 checkpoint 경로도 함께 변경해야 한다.

DnCNN 사용 시:

```python
checkpoint_path = "../dncnn/best_model.pt"
```

Transformer 사용 시:

```python
checkpoint_path = "../transformer/best_model.pt"
```

즉, 전체 파이프라인 구조는 유지하고, 모델 생성 함수와 checkpoint 경로만 교체하면 된다.

---

## 10. Checkpoint 로드 방식

현재 파이프라인은 PyTorch의 `state_dict` 기반 checkpoint를 사용한다.

학습 코드에서는 다음과 같이 weight를 저장한다.

```python
torch.save(model.state_dict(), "best_model.pt")
```

파이프라인에서는 먼저 모델 구조를 생성한 뒤, 저장된 weight를 로드한다.

```python
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
```

이 방식은 모델 전체를 저장하는 방식보다 GitHub 공유 및 협업에 적합하다. 모델 구조는 코드에 명시되고, checkpoint 파일에는 학습된 weight만 저장되기 때문이다.

---

## 11. 현재 구현상의 참고 사항

현재는 DnCNN 모델 클래스가 `denoising_pipeline.py` 내부에 직접 포함되어 있다.

이는 임시 구조이며, 추후 코드 정리 단계에서는 모델 구조를 별도 파일로 분리하는 것이 좋다.

예상 구조는 다음과 같다.

```text
denoiser/
├── dncnn/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── best_model.pt
│
├── transformer/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── best_model.pt
│
└── pipeline/
    ├── denoising_pipeline.py
    └── README.md
```

이렇게 분리하면 `denoising_pipeline.py`에서는 필요한 모델만 import하여 사용할 수 있다.

---

## 12. 추후 개선 예정

현재 파이프라인은 단일 이미지 기반으로 동작한다.

향후 추가 가능한 기능은 다음과 같다.

- 여러 이미지에 대한 배치 처리
- 영상 파일 처리
- 모델 선택 옵션 추가
- 결과 저장 폴더 자동 생성
- DnCNN / Transformer 모델 선택 기능
- Super-Resolution 모듈과 연동
- OCR 모듈과 연동
- 모델 구조 외부 모듈화
- 결과 비교 이미지 자동 생성

---

## 13. 주의 사항

- `output.png`는 실행 시 자동으로 생성된다.
- `output.png`가 이미 존재하는 경우 새 결과로 덮어쓰기된다.
- 현재는 Grayscale 이미지 기준으로 동작한다.
- 입력 이미지는 내부적으로 모델 입력 크기에 맞게 resize된다.
- 결과 이미지는 후처리 과정에서 원본 해상도로 다시 resize되어 저장된다.
- 현재 기본 모델은 DnCNN이다.
- 추후 Transformer 기반 디노이저로 교체할 수 있다.
