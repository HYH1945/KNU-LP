import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# 설정
class Config:
    test_root = "../data/dataset/test"
    checkpoint_path = "best_model.pt"

    image_size = (48, 128)   # (H, W)
    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transformer 관련
    embed_dim = 64
    num_heads = 4
    depth = 4
    mlp_ratio = 4.0
    dropout = 0.1

    sample_count = 3
    pixel_acc_threshold = 0.05


# 데이터셋
class PlateDenoiseDataset(Dataset):

    def __init__(self, split_root: str, image_size=(48, 128)):
        self.clean_dir = Path(split_root) / "clean"
        self.noisy_dir = Path(split_root) / "noisy"
        self.image_size = image_size

        if not self.clean_dir.exists():
            raise FileNotFoundError(f"clean 폴더 없음: {self.clean_dir}")
        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"noisy 폴더 없음: {self.noisy_dir}")

        clean_files = sorted([f.name for f in self.clean_dir.iterdir() if f.is_file()])
        noisy_files = set([f.name for f in self.noisy_dir.iterdir() if f.is_file()])

        self.file_list = [f for f in clean_files if f in noisy_files]

        if len(self.file_list) == 0:
            raise ValueError("짝이 맞는 이미지가 없습니다.")

    def __len__(self):
        return len(self.file_list)

    def _load_gray(self, path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지 읽기 실패: {path}")

        h, w = self.image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # [1, H, W]
        return torch.tensor(img, dtype=torch.float32)

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        clean_path = self.clean_dir / fname
        noisy_path = self.noisy_dir / fname

        target = self._load_gray(clean_path)
        inp = self._load_gray(noisy_path)

        return {
            "filename": fname,
            "input": inp,
            "target": target
        }


# 평가 지표
def psnr_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred_mean = pred.mean().item()
    target_mean = target.mean().item()

    pred_var = pred.var(unbiased=False).item()
    target_var = target.var(unbiased=False).item()

    covariance = ((pred - pred.mean()) * (target - target.mean())).mean().item()

    numerator = (2 * pred_mean * target_mean + C1) * (2 * covariance + C2)
    denominator = (pred_mean ** 2 + target_mean ** 2 + C1) * (pred_var + target_var + C2)

    return numerator / (denominator + 1e-8)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> float:
    diff = torch.abs(pred - target)
    return (diff <= threshold).float().mean().item()


# Transformer 블록
class PositionalEncoding2D(nn.Module):

    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


class TransformerDenoiser(nn.Module):

    def __init__(
        self,
        image_size=(48, 128),
        in_channels=1,
        embed_dim=64,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.h, self.w = image_size
        self.embed_dim = embed_dim
        self.num_tokens = self.h * self.w

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

        self.pos_encoding = PositionalEncoding2D(self.num_tokens, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        feat = self.embed(x)                         # [B, D, H, W]
        tokens = feat.flatten(2).transpose(1, 2)    # [B, H*W, D]
        tokens = self.pos_encoding(tokens)
        tokens = self.encoder(tokens)

        feat_out = tokens.transpose(1, 2).reshape(b, self.embed_dim, h, w)
        noise_or_residual = self.reconstruct(feat_out)

        out = x + noise_or_residual
        out = torch.clamp(out, 0.0, 1.0)
        return out


# 평가
@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold=0.05):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0

    for batch in dataloader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        pred = model(x)

        l1_val = F.l1_loss(pred, y).item()
        mae_val = torch.mean(torch.abs(pred - y)).item()
        mse_val = F.mse_loss(pred, y).item()
        psnr_val = psnr_torch(pred, y)
        ssim_val = ssim_torch(pred, y)
        acc_val = pixel_accuracy(pred, y, threshold=threshold)

        total_loss += l1_val
        total_mae += mae_val
        total_mse += mse_val
        total_psnr += psnr_val
        total_ssim += ssim_val
        total_acc += acc_val

    n = len(dataloader)

    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "mse": total_mse / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "acc": total_acc / n,
    }


def print_eval_results(results, title="Transformer Evaluation Results"):
    print(f"\n===== {title} =====")
    print(f"Loss : {results['loss']:.6f}")
    print(f"MAE  : {results['mae']:.6f}")
    print(f"MSE  : {results['mse']:.6f}")
    print(f"PSNR : {results['psnr']:.4f}")
    print(f"SSIM : {results['ssim']:.4f}")
    print(f"ACC  : {results['acc']:.4f}")


# 시각화
@torch.no_grad()
def show_samples(model, dataloader, device, num_samples=3):
    model.eval()

    batch = next(iter(dataloader))
    x = batch["input"].to(device)
    y = batch["target"].to(device)
    filenames = batch["filename"]

    pred = model(x)

    num_samples = min(num_samples, x.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        inp_img = x[i].cpu().squeeze(0).numpy()
        pred_img = pred[i].cpu().squeeze(0).numpy()
        target_img = y[i].cpu().squeeze(0).numpy()

        axes[i, 0].imshow(inp_img, cmap="gray")
        axes[i, 0].set_title(f"Noisy Input\n{filenames[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_img, cmap="gray")
        axes[i, 1].set_title("Transformer Output")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(target_img, cmap="gray")
        axes[i, 2].set_title("Target (GT)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


# 체크포인트 로드
def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


# 메인
def main():
    cfg = Config()

    print(f"사용 디바이스: {cfg.device}")

    test_set = PlateDenoiseDataset(
        split_root=cfg.test_root,
        image_size=cfg.image_size
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    print(f"Test 수: {len(test_set)}")

    model = TransformerDenoiser(
        image_size=cfg.image_size,
        in_channels=1,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        depth=cfg.depth,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(cfg.device)

    model = load_checkpoint(model, cfg.checkpoint_path, cfg.device)

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=cfg.device,
        threshold=cfg.pixel_acc_threshold
    )
    print_eval_results(results, title="Best Transformer Test Results")

    show_samples(model, test_loader, cfg.device, num_samples=cfg.sample_count)


if __name__ == "__main__":
    main()