# Denoising Pipeline

## 1. 개요
본 모듈은 단일 이미지 입력을 받아 디노이징 모델을 통과시킨 뒤,
결과 이미지를 저장하는 파이프라인이다.

모델 구조는 DnCNN, Transformer 등으로 교체할 수 있으며,
전처리 / 추론 / 후처리 / 저장 과정을 하나의 흐름으로 관리한다.

---

## 2. 기본 실행 방식

같은 폴더 안에 `input.png`를 준비한 뒤 다음 명령어를 실행한다.

```bash
python inference_pipeline.py