from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# 설정
class Config:
    input_path = "input.png"
    output_path = "output.png"

    image_size = (48, 128)  # (H, W)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 실제 모델 경로 (현재 dncnn 모델 기준)
    checkpoint_path = "../dncnn/best_model.pt"


# 임시
class IdentityModel(nn.Module):

    def forward(self, x):
        return x


class DnCNN(nn.Module):
    """
    DnCNN 기반 디노이저

    핵심:
    - 모델이 노이즈를 예측
    - 입력 이미지에서 예측한 노이즈를 빼서 복원 이미지 생성
    """

    def __init__(self, in_channels=1, depth=17, features=64):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, in_channels, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        denoised = x - noise
        denoised = torch.clamp(denoised, 0.0, 1.0)
        return denoised



# Pipeline
class ImageDenoisingPipeline:
    """
    이미지 입력 -> 전처리 -> 모델 추론 -> 후처리 -> 결과 저장
    """

    def __init__(self, model, device="cpu", image_size=(48, 128)):
        self.model = model.to(device)
        self.device = device
        self.image_size = image_size
        self.model.eval()

    def load_checkpoint(self, checkpoint_path):
        """
        state_dict 방식의 checkpoint 로드
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, image_path):
        """
        입력 이미지를 모델 입력 tensor로 변환
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"이미지 읽기 실패: {image_path}")

        original_shape = img.shape  # (H, W)

        h, w = self.image_size
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.tensor(normalized, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        return tensor.to(self.device), original_shape

    def postprocess(self, output_tensor, original_shape):
        """
        모델 출력 tensor를 이미지로 변환
        """
        output = output_tensor.squeeze().detach().cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        original_h, original_w = original_shape
        output = cv2.resize(
            output,
            (original_w, original_h),
            interpolation=cv2.INTER_CUBIC
        )

        return output

    @torch.no_grad()
    def run(self, input_path, output_path=None, return_image=True):
        """
        단일 이미지 추론 실행
        """
        input_path = Path(input_path)

        x, original_shape = self.preprocess(input_path)
        pred = self.model(x)
        output_img = self.postprocess(pred, original_shape)

        if output_path is not None:
            cv2.imwrite(str(output_path), output_img)
            print(f"[INFO] output saved: {output_path}")

        if return_image:
            return output_img

        return None


# 모델 생성
def build_model():
    """
    현재는 위에서 정의한 DnCNN 모델 사용
    추후 model.py 로 모델 구조를 분리 후 import 방식으로 교체 예정
    """
    return DnCNN(in_channels=1, depth=17, features=64)


def main():
    cfg = Config()

    model = build_model()

    pipeline = ImageDenoisingPipeline(
        model=model,
        device=cfg.device,
        image_size=cfg.image_size
    )

    if cfg.checkpoint_path is not None:
        pipeline.load_checkpoint(cfg.checkpoint_path)

    pipeline.run(
        input_path=cfg.input_path,
        output_path=cfg.output_path,
        return_image=False
    )


if __name__ == "__main__":
    main()