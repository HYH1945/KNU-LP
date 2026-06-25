import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Config:
    train_root = "../data/dataset/train"
    val_root = "../data/dataset/val"

    image_size = (48, 128)
    batch_size = 32
    epochs = 30
    lr = 1e-4
    seed = 42
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "best_model.pt"

    # Mixed objective:
    # 1) Denoise path: noisy -> low-noise target
    # 2) Identity path: low-noise -> low-noise target
    identity_ratio = 0.4
    denoise_loss_weight = 1.0
    identity_loss_weight = 1.0


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    pred_mean = pred.mean().item()
    target_mean = target.mean().item()

    pred_var = pred.var(unbiased=False).item()
    target_var = target.var(unbiased=False).item()

    covariance = ((pred - pred.mean()) * (target - target.mean())).mean().item()

    numerator = (2 * pred_mean * target_mean + c1) * (2 * covariance + c2)
    denominator = (pred_mean ** 2 + target_mean ** 2 + c1) * (pred_var + target_var + c2)

    return numerator / (denominator + 1e-8)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> float:
    diff = torch.abs(pred - target)
    return (diff <= threshold).float().mean().item()


class PlateDenoiseDataset(Dataset):
    def __init__(self, split_root: str, image_size=(48, 128)):
        self.clean_dir = Path(split_root) / "clean"
        self.noisy_dir = Path(split_root) / "noisy"
        self.image_size = image_size

        if not self.clean_dir.exists():
            raise FileNotFoundError(f"clean directory not found: {self.clean_dir}")
        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"noisy directory not found: {self.noisy_dir}")

        clean_files = sorted([f.name for f in self.clean_dir.iterdir() if f.is_file()])
        noisy_files = set([f.name for f in self.noisy_dir.iterdir() if f.is_file()])

        self.file_list = [f for f in clean_files if f in noisy_files]
        if not self.file_list:
            raise ValueError("No matching filename pairs between clean and noisy directories.")

    def __len__(self):
        return len(self.file_list)

    def _load_gray(self, path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        h, w = self.image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
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
            "target": target,
            "mode": "denoise",
        }


class MixedPlateDenoiseDataset(PlateDenoiseDataset):
    def __init__(self, split_root: str, image_size=(48, 128), identity_ratio=0.4):
        super().__init__(split_root=split_root, image_size=image_size)
        self.identity_ratio = float(identity_ratio)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # Identity branch: low-noise input should pass through unchanged.
        if random.random() < self.identity_ratio:
            sample["input"] = sample["target"].clone()
            sample["mode"] = "identity"

        return sample


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

    def forward(self, x):
        noise = self.net(x)
        denoised = x - noise
        return torch.clamp(denoised, 0.0, 1.0)


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
        axes[i, 0].set_title(f"Input\\n{filenames[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_img, cmap="gray")
        axes[i, 1].set_title("DnCNN Output")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(target_img, cmap="gray")
        axes[i, 2].set_title("Target")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def _compute_weighted_l1(pred, target, mode_list, config, device):
    denoise_mask = torch.tensor([m == "denoise" for m in mode_list], device=device, dtype=torch.bool)
    identity_mask = ~denoise_mask

    denoise_loss = torch.tensor(0.0, device=device)
    identity_loss = torch.tensor(0.0, device=device)

    if denoise_mask.any():
        denoise_loss = F.l1_loss(pred[denoise_mask], target[denoise_mask])
    if identity_mask.any():
        identity_loss = F.l1_loss(pred[identity_mask], target[identity_mask])

    total_loss = (
        config.denoise_loss_weight * denoise_loss
        + config.identity_loss_weight * identity_loss
    )

    return total_loss, denoise_loss, identity_loss


def train_one_epoch(model, dataloader, optimizer, device, config):
    model.train()

    total_loss = 0.0
    total_denoise_loss = 0.0
    total_identity_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0

    for batch in dataloader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        mode = batch.get("mode", ["denoise"] * x.size(0))

        pred = model(x)
        loss, denoise_loss, identity_loss = _compute_weighted_l1(pred, y, mode, config, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_denoise_loss += denoise_loss.item()
        total_identity_loss += identity_loss.item()
        total_psnr += psnr_torch(pred.detach(), y)
        total_ssim += ssim_torch(pred.detach(), y)
        total_acc += pixel_accuracy(pred.detach(), y)

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "denoise_loss": total_denoise_loss / n,
        "identity_loss": total_identity_loss / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "acc": total_acc / n,
    }


@torch.no_grad()
def validate_denoise(model, dataloader, device):
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


@torch.no_grad()
def validate_identity(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0

    for batch in dataloader:
        y = batch["target"].to(device)
        x = y

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


def train_model(model, train_loader, val_loader, identity_val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_score = float("inf")
    train_history = {"loss": [], "denoise_loss": [], "identity_loss": [], "psnr": [], "ssim": [], "acc": []}
    val_history = {
        "denoise_loss": [],
        "identity_loss": [],
        "score": [],
        "psnr": [],
        "ssim": [],
        "acc": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config.device, config)
        val_denoise = validate_denoise(model, val_loader, config.device)
        val_identity = validate_identity(model, identity_val_loader, config.device)

        val_score = (
            config.denoise_loss_weight * val_denoise["loss"]
            + config.identity_loss_weight * val_identity["loss"]
        )

        for k in train_history:
            train_history[k].append(train_metrics[k])

        val_history["denoise_loss"].append(val_denoise["loss"])
        val_history["identity_loss"].append(val_identity["loss"])
        val_history["score"].append(val_score)
        val_history["psnr"].append(val_denoise["psnr"])
        val_history["ssim"].append(val_denoise["ssim"])
        val_history["acc"].append(val_denoise["acc"])

        print(
            f"[Epoch {epoch:02d}/{config.epochs}] "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"(D={train_metrics['denoise_loss']:.4f}, I={train_metrics['identity_loss']:.4f}) | "
            f"Val Denoise={val_denoise['loss']:.4f}, "
            f"Val Identity={val_identity['loss']:.4f}, "
            f"Val Score={val_score:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), config.save_path)
            print(f"  -> best model saved: {config.save_path}")

    return train_history, val_history


def plot_history(train_history, val_history):
    plot_pairs = [
        ("loss", "score"),
        ("denoise_loss", "denoise_loss"),
        ("identity_loss", "identity_loss"),
        ("psnr", "psnr"),
        ("ssim", "ssim"),
        ("acc", "acc"),
    ]

    for train_key, val_key in plot_pairs:
        plt.figure(figsize=(6, 4))
        plt.plot(train_history[train_key], label=f"train_{train_key}")
        plt.plot(val_history[val_key], label=f"val_{val_key}")
        plt.xlabel("Epoch")
        plt.ylabel(train_key.upper())
        plt.title(f"{train_key.upper()} vs {val_key.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    cfg = Config()
    set_seed(cfg.seed)

    train_set = MixedPlateDenoiseDataset(
        split_root=cfg.train_root,
        image_size=cfg.image_size,
        identity_ratio=cfg.identity_ratio,
    )
    val_set = PlateDenoiseDataset(split_root=cfg.val_root, image_size=cfg.image_size)

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    print(
        "Mixed training config | "
        f"identity_ratio={cfg.identity_ratio:.2f}, "
        f"denoise_w={cfg.denoise_loss_weight:.2f}, "
        f"identity_w={cfg.identity_loss_weight:.2f}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    identity_val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = DnCNN(in_channels=1, depth=17, features=64).to(cfg.device)
    print(f"Device: {cfg.device}")

    train_history, val_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        identity_val_loader=identity_val_loader,
        config=cfg,
    )

    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))

    plot_history(train_history, val_history)
    show_samples(model, val_loader, cfg.device, num_samples=3)


if __name__ == "__main__":
    main()
