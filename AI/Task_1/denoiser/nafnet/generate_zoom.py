import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import NAFNet


BASE_DIR = Path(__file__).resolve().parent


def load_state_dict(weight_path, device):
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def build_model(device, weight_path):
    model = NAFNet(
        in_channels=3,
        out_channels=3,
        width=32,
        enc_blk_nums=[1, 1, 1, 14],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )
    model.load_state_dict(load_state_dict(weight_path, device))
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate a zoomed NAFNet denoising comparison.")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR.parent / "data" / "dataset_color"))
    parser.add_argument(
        "--weight_path",
        type=str,
        default=str(BASE_DIR / "checkpoints_nafnet" / "best_nafnet_model_ratio0.pt"),
    )
    parser.add_argument("--output_dir", type=str, default=str(BASE_DIR / "results" / "nafnet_samples"))
    parser.add_argument("--sample", type=str, default="0002.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, Path(args.weight_path))

    noisy_path = Path(args.data_dir) / "test" / "noisy" / args.sample
    noisy = cv2.imread(str(noisy_path))
    if noisy is None:
        raise FileNotFoundError(f"Sample image not found: {noisy_path}")

    rgb_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    tensor_noisy = (
        torch.from_numpy(rgb_noisy.astype(np.float32).transpose(2, 0, 1) / 255.0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        tensor_denoised = torch.clamp(model(tensor_noisy), 0, 1)

    rgb_denoised = (
        tensor_denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
    ).astype(np.uint8)
    denoised_img = cv2.cvtColor(rgb_denoised, cv2.COLOR_RGB2BGR)

    crop_n = noisy[10:30, 45:65]
    crop_d = denoised_img[10:30, 45:65]

    crop_n = cv2.resize(crop_n, (200, 200), interpolation=cv2.INTER_NEAREST)
    crop_d = cv2.resize(crop_d, (200, 200), interpolation=cv2.INTER_NEAREST)

    h, w = crop_n.shape[:2]
    mosaic = np.zeros((h, w * 2, 3), dtype=np.uint8)
    mosaic[:, :w] = crop_n
    mosaic[:, w:] = crop_d

    cv2.putText(mosaic, "Noisy (Zoomed)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(mosaic, "Denoised (Zoomed)", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "zoom_proof.png"
    cv2.imwrite(str(save_path), mosaic)
    print(f"Generated {save_path}")


if __name__ == "__main__":
    main()
