"""
DnCNN
- 노이즈 잔차를 예측한 뒤 입력에서 빼서 복원 이미지를 산출하는 디노이저 모델
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN 기반 디노이저

    핵심:
    - 모델이 노이즈를 예측
    - 입력 이미지에서 예측한 노이즈를 빼서 복원 이미지 생성
    """

    def __init__(self, in_channels: int = 1, depth: int = 17, features: int = 64):
        """
        Input:
            in_channels (int) : 입력 채널 수 (grayscale=1)
            depth       (int) : 컨볼루션 레이어 깊이
            features    (int) : 중간 feature map 채널 수
        """
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

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x (torch.Tensor) : 입력 텐서 (B, in_channels, H, W), float 0~1
        Output:
            denoised (torch.Tensor) : 디노이즈된 텐서 (B, in_channels, H, W), float 0~1
        """
        noise = self.net(x)
        denoised = x - noise
        denoised = torch.clamp(denoised, 0.0, 1.0)
        return denoised


# ====================================================================== #
if __name__ == "__main__":
    """
    단독 실행 테스트:
    - 더미 텐서로 forward 통과 + shape/range 검증
    """
    model = DnCNN(in_channels=1, depth=17, features=64)
    model.eval()

    dummy = torch.rand(1, 1, 48, 128)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == dummy.shape, f"출력 shape 오류: {out.shape}"
    assert (0.0 <= out).all() and (out <= 1.0).all(), "출력 범위 오류 (0~1 벗어남)"

    print(f"✅ DnCNN 단독 테스트 통과 | shape={tuple(out.shape)}")
