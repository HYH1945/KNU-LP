import os
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

    train_root = "../data/dataset/train"
    val_root = "../data/dataset/val"

    image_size = (48, 128)   # (H, W)
    batch_size = 32
    epochs = 30
    lr = 1e-4
    seed = 42
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "best_model.pt"

# 유틸
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# PSNR - 값이 높을수록 GT와 가까움
def psnr_torch(pred: torch.Tensor, target: torch.Tensor) -> float:

    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)

# 전역 SSIM - 학습 모니터링용
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

# 보조 지표
def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> float:

    diff = torch.abs(pred - target)
    correct = (diff <= threshold).float().mean().item()
    return correct


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

        # 같은 이름만 사용
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

        target = self._load_gray(clean_path)    # GT
        inp = self._load_gray(noisy_path)       # 입력

        return {
            "filename": fname,
            "input": inp,
            "target": target
        }


# DnCNN 모델
class DnCNN(nn.Module):

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


# 시각화
def show_samples(model, dataloader, device, num_samples=3):

    model.eval()
    batch = next(iter(dataloader))

    x = batch["input"].to(device)
    y = batch["target"].to(device)
    filenames = batch["filename"]

    with torch.no_grad():
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
        axes[i, 1].set_title("DnCNN Output")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(target_img, cmap="gray")
        axes[i, 2].set_title("Target (GT)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


# 학습 / 검증
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0

    for batch in dataloader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        pred = model(x)
        loss = F.l1_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr_torch(pred.detach(), y)
        total_ssim += ssim_torch(pred.detach(), y)
        total_acc += pixel_accuracy(pred.detach(), y)

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "acc": total_acc / n,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0

    for batch in dataloader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        pred = model(x)
        loss = F.l1_loss(pred, y)

        total_loss += loss.item()
        total_psnr += psnr_torch(pred, y)
        total_ssim += ssim_torch(pred, y)
        total_acc += pixel_accuracy(pred, y)

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "acc": total_acc / n,
    }


def train_model(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    train_history = {"loss": [], "psnr": [], "ssim": [], "acc": []}
    val_history = {"loss": [], "psnr": [], "ssim": [], "acc": []}

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config.device)
        val_metrics = validate(model, val_loader, config.device)

        for k in train_history:
            train_history[k].append(train_metrics[k])
            val_history[k].append(val_metrics[k])

        print(
            f"[Epoch {epoch:02d}/{config.epochs}] "
            f"Train Loss={train_metrics['loss']:.4f}, PSNR={train_metrics['psnr']:.2f}, SSIM={train_metrics['ssim']:.4f}, Acc={train_metrics['acc']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.4f}, Acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), config.save_path)
            print(f"  -> best model saved: {config.save_path}")

    return train_history, val_history


# 그래프 출력
def plot_history(train_history, val_history):
    metrics = ["loss", "psnr", "ssim", "acc"]

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        plt.plot(train_history[metric], label=f"train_{metric}")
        plt.plot(val_history[metric], label=f"val_{metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.title(metric.upper())
        plt.legend()
        plt.tight_layout()
        plt.show()


# 실행
def main():
    cfg = Config()
    set_seed(cfg.seed)

    # 1) train / val 데이터셋 로딩
    train_set = PlateDenoiseDataset(
        split_root=cfg.train_root,
        image_size=cfg.image_size
    )

    val_set = PlateDenoiseDataset(
        split_root=cfg.val_root,
        image_size=cfg.image_size
    )

    print(f"Train 수: {len(train_set)}")
    print(f"Val 수: {len(val_set)}")

    # 2) DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # 3) 모델 생성
    model = DnCNN(in_channels=1, depth=17, features=64).to(cfg.device)
    print(f"사용 디바이스: {cfg.device}")

    # 4) 학습
    train_history, val_history = train_model(model, train_loader, val_loader, cfg)

    # 5) best 모델 로드
    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))

    # 6) 그래프 출력
    plot_history(train_history, val_history)

    # 7) 샘플 시각화
    show_samples(model, val_loader, cfg.device, num_samples=3)


if __name__ == "__main__":
    main()