import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.metrics import pixel_accuracy_torch as pixel_accuracy
from utils.metrics import psnr_torch, ssim_torch
from utils.visualize import show_samples_torch as show_samples


class Config:
    test_root = "../data/dataset/test"
    checkpoint_path = "best_model.pt"

    image_size = (48, 128)
    batch_size = 32
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_count = 3
    pixel_acc_threshold = 0.05

    # Mixed ratio means: fraction of clean-like(identity) inputs in evaluation batch stream.
    mixed_clean_ratios = [0.2, 0.5, 0.8]
    mixed_seed = 42


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
            raise ValueError("No matched clean/noisy image pairs.")

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


class CleanDataset(PlateDenoiseDataset):
    def __init__(self, split_root: str, image_size=(48, 128)):
        super().__init__(split_root=split_root, image_size=image_size)
        self.noisy_dir = self.clean_dir

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["mode"] = "identity"
        return sample


class MixedInputDataset(Dataset):
    def __init__(self, base_dataset: PlateDenoiseDataset, clean_ratio: float, seed: int = 42):
        self.base = base_dataset
        self.clean_ratio = float(clean_ratio)

        rng = random.Random(seed)
        self.identity_flags = [rng.random() < self.clean_ratio for _ in range(len(self.base))]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self.identity_flags[idx]:
            sample["input"] = sample["target"].clone()
            sample["mode"] = "identity"
        else:
            sample["mode"] = "denoise"
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


@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold=0.05):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0
    denoise_count = 0
    identity_count = 0

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

        mode = batch.get("mode")
        if mode is not None:
            denoise_count += sum(m == "denoise" for m in mode)
            identity_count += sum(m == "identity" for m in mode)

    n = len(dataloader)

    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "mse": total_mse / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "acc": total_acc / n,
        "denoise_count": denoise_count,
        "identity_count": identity_count,
    }


def print_eval_results(results, title="DnCNN Evaluation Results"):
    print(f"\n===== {title} =====")
    print(f"Loss : {results['loss']:.6f}")
    print(f"MAE  : {results['mae']:.6f}")
    print(f"MSE  : {results['mse']:.6f}")
    print(f"PSNR : {results['psnr']:.4f}")
    print(f"SSIM : {results['ssim']:.4f}")
    print(f"ACC  : {results['acc']:.4f}")

    if results.get("denoise_count", 0) or results.get("identity_count", 0):
        total = results.get("denoise_count", 0) + results.get("identity_count", 0)
        if total > 0:
            clean_ratio = results.get("identity_count", 0) / total
            print(
                f"Input Mix: clean-like={results.get('identity_count', 0)} "
                f"({clean_ratio * 100:.1f}%), noisy-like={results.get('denoise_count', 0)}"
            )


def _to_float_results(metrics: dict):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# Backward-compatible alias used by existing scripts.
def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def evaluate_mixed_ratios(model, base_dataset, cfg):
    print("\n================ Mixed-Ratio Sensitivity ================")
    mixed_map = {}

    for ratio in cfg.mixed_clean_ratios:
        mixed_ds = MixedInputDataset(base_dataset, clean_ratio=ratio, seed=cfg.mixed_seed)
        mixed_loader = DataLoader(
            mixed_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        results = evaluate_model(
            model=model,
            dataloader=mixed_loader,
            device=cfg.device,
            threshold=cfg.pixel_acc_threshold,
        )
        print_eval_results(results, title=f"Mixed Eval (clean ratio={ratio:.1f})")
        mixed_map[f"{ratio:.1f}"] = _to_float_results(results)

    return mixed_map


def _append_summary_csv(csv_path: Path, timestamp: str, model_name: str, scenario: str, metrics: dict, cfg: Config):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "model",
        "scenario",
        "loss",
        "mae",
        "mse",
        "psnr",
        "ssim",
        "acc",
        "denoise_count",
        "identity_count",
        "device",
        "batch_size",
        "checkpoint_path",
    ]
    row = {
        "timestamp": timestamp,
        "model": model_name,
        "scenario": scenario,
        "loss": metrics.get("loss"),
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "acc": metrics.get("acc"),
        "denoise_count": metrics.get("denoise_count"),
        "identity_count": metrics.get("identity_count"),
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "checkpoint_path": cfg.checkpoint_path,
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_eval_artifacts(base_dir: Path, run_name: str, payload: dict, cfg: Config):
    base_dir.mkdir(parents=True, exist_ok=True)
    json_path = base_dir / f"{run_name}.json"
    csv_path = base_dir / "dncnn_eval_summary.csv"
    log_path = base_dir / f"{run_name}.log"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Device: {cfg.device}\n")
        f.write(f"Checkpoint: {cfg.checkpoint_path}\n")
        f.write(f"Test root: {cfg.test_root}\n")
        f.write(f"Batch size: {cfg.batch_size}\n")
        f.write(f"Threshold: {cfg.pixel_acc_threshold}\n")
    return json_path, csv_path, log_path


def parse_args():
    parser = argparse.ArgumentParser(description="DnCNN evaluation")
    parser.add_argument("--save", action="store_true", help="Save evaluation outputs (json/csv/log).")
    parser.add_argument("--save-dir", default="../results", help="Directory to store outputs.")
    parser.add_argument("--run-name", default=None, help="Optional run name prefix.")
    parser.add_argument("--skip-mixed", action="store_true", help="Skip mixed-ratio evaluation.")
    parser.add_argument("--no-show", action="store_true", help="Disable sample visualization.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    print(f"Device: {cfg.device}")

    test_set = PlateDenoiseDataset(split_root=cfg.test_root, image_size=cfg.image_size)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    clean_test_set = CleanDataset(split_root=cfg.test_root, image_size=cfg.image_size)
    clean_test_loader = DataLoader(
        clean_test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    print(f"Test samples: {len(test_set)}")

    model = DnCNN(in_channels=1, depth=17, features=64).to(cfg.device)
    model = load_checkpoint(model, cfg.checkpoint_path, cfg.device)

    noisy_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=cfg.device,
        threshold=cfg.pixel_acc_threshold,
    )
    print_eval_results(noisy_results, title="DnCNN Noisy-Only Evaluation")

    identity_results = evaluate_model(
        model=model,
        dataloader=clean_test_loader,
        device=cfg.device,
        threshold=cfg.pixel_acc_threshold,
    )
    print_eval_results(identity_results, title="DnCNN Identity Evaluation")

    mixed_results = {}
    if not args.skip_mixed:
        mixed_results = evaluate_mixed_ratios(model, test_set, cfg)

    if args.save:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_name = args.run_name or f"dncnn_eval_{timestamp}"
        payload = {
            "run_name": run_name,
            "timestamp": timestamp,
            "model": "DnCNN",
            "config": {
                "device": cfg.device,
                "batch_size": cfg.batch_size,
                "test_root": cfg.test_root,
                "checkpoint_path": cfg.checkpoint_path,
                "image_size": list(cfg.image_size),
                "pixel_acc_threshold": cfg.pixel_acc_threshold,
                "mixed_clean_ratios": cfg.mixed_clean_ratios,
            },
            "results": {
                "noisy_only": _to_float_results(noisy_results),
                "identity": _to_float_results(identity_results),
                "mixed": mixed_results,
            },
        }
        json_path, csv_path, log_path = _save_eval_artifacts(Path(args.save_dir), run_name, payload, cfg)
        _append_summary_csv(csv_path, timestamp, "DnCNN", "noisy_only", payload["results"]["noisy_only"], cfg)
        _append_summary_csv(csv_path, timestamp, "DnCNN", "identity", payload["results"]["identity"], cfg)
        for ratio, metrics in payload["results"]["mixed"].items():
            _append_summary_csv(csv_path, timestamp, "DnCNN", f"mixed_clean_{ratio}", metrics, cfg)
        print(f"\nSaved JSON: {json_path}")
        print(f"Saved CSV : {csv_path}")
        print(f"Saved LOG : {log_path}")

    if not args.no_show:
        show_samples(model, test_loader, cfg.device, num_samples=cfg.sample_count)


if __name__ == "__main__":
    main()
