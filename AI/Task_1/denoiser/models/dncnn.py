import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels=1, depth=17, features=64):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, 3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.append(nn.Conv2d(features, in_channels, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

        # Kaiming Normal 가중치 초기화 추가
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        noise = self.net(x)
        denoised = x - noise
        return torch.clamp(denoised, 0.0, 1.0)
