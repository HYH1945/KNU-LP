import argparse
import os
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


def build_model(device):
    model = NAFNet(
        in_channels=3,
        out_channels=3,
        width=32,
        enc_blk_nums=[1, 1, 1, 14],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate NAFNet process visualization samples.")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR.parent / "data" / "dataset_color"))
    parser.add_argument(
        "--weight_path",
        type=str,
        default=str(BASE_DIR / "checkpoints_nafnet" / "best_nafnet_model_ratio0.pt"),
    )
    parser.add_argument("--output_dir", type=str, default=str(BASE_DIR / "results" / "nafnet_samples"))
    parser.add_argument("--samples", nargs="+", default=["0001.png", "0002.png", "0004.png"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    model.load_state_dict(load_state_dict(args.weight_path, device))

    noisy_dir = Path(args.data_dir) / "test" / "noisy"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname in args.samples:
        noisy_img = cv2.imread(str(noisy_dir / fname))
        if noisy_img is None:
            print(f"Skip missing sample: {noisy_dir / fname}")
            continue

        rgb_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
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

        res_img = denoised_img.astype(np.float32) - noisy_img.astype(np.float32)
        res_vis = np.clip((res_img * 2.0) + 128, 0, 255).astype(np.uint8)

        h, w = noisy_img.shape[:2]
        process_mosaic = np.zeros((h, w * 3, 3), dtype=np.uint8)
        process_mosaic[:, :w] = noisy_img
        process_mosaic[:, w : w * 2] = res_vis
        process_mosaic[:, w * 2 :] = denoised_img

        cv2.putText(process_mosaic, "Input (Noisy)", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(process_mosaic, "Predicted Residual", (w + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(process_mosaic, "Output (Denoised)", (w * 2 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        save_path = output_dir / f"denoise_process_{fname}"
        cv2.imwrite(str(save_path), process_mosaic)
        print(f"Generated {save_path}")


if __name__ == "__main__":
    main()
