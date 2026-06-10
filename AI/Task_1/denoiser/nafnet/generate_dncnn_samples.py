import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
BE_ROOT = REPO_ROOT / "BE"

if str(BE_ROOT) not in sys.path:
    sys.path.append(str(BE_ROOT))

from pipeline.denoiser.model import DnCNN


def load_state_dict(weight_path, device):
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def build_model(device, weight_path):
    model = DnCNN(in_channels=3, depth=17, features=64)
    model.load_state_dict(load_state_dict(weight_path, device))
    model.to(device)
    model.eval()
    return model


def process_and_save(model, device, noisy_dir, output_dir, fname, is_zoom=False):
    noisy_path = noisy_dir / fname
    noisy_img = cv2.imread(str(noisy_path))
    if noisy_img is None:
        print(f"Skip missing sample: {noisy_path}")
        return

    rgb_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    tensor_noisy = (
        torch.from_numpy(rgb_noisy.astype(np.float32).transpose(2, 0, 1) / 255.0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        tensor_denoised = model(tensor_noisy)

    rgb_denoised = (
        tensor_denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
    ).astype(np.uint8)
    denoised_img = cv2.cvtColor(rgb_denoised, cv2.COLOR_RGB2BGR)

    if not is_zoom:
        res_img = noisy_img.astype(np.float32) - denoised_img.astype(np.float32)
        res_vis = np.clip((res_img * 2.0) + 128, 0, 255).astype(np.uint8)

        h, w = noisy_img.shape[:2]
        mosaic = np.zeros((h, w * 3, 3), dtype=np.uint8)
        mosaic[:, :w] = noisy_img
        mosaic[:, w : w * 2] = res_vis
        mosaic[:, w * 2 :] = denoised_img

        cv2.putText(mosaic, "Input (Noisy)", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(mosaic, "Predicted Residual", (w + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(mosaic, "Output (DnCNN)", (w * 2 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        save_path = output_dir / f"dncnn_process_{fname}"
        cv2.imwrite(str(save_path), mosaic)
        print(f"Generated {save_path}")
        return

    crop_n = noisy_img[10:30, 45:65]
    crop_d = denoised_img[10:30, 45:65]

    crop_n = cv2.resize(crop_n, (200, 200), interpolation=cv2.INTER_NEAREST)
    crop_d = cv2.resize(crop_d, (200, 200), interpolation=cv2.INTER_NEAREST)

    h, w = crop_n.shape[:2]
    mosaic = np.zeros((h, w * 2, 3), dtype=np.uint8)
    mosaic[:, :w] = crop_n
    mosaic[:, w:] = crop_d

    cv2.putText(mosaic, "Noisy (Zoomed)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(mosaic, "DnCNN (Zoomed)", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    save_path = output_dir / "dncnn_zoom_proof.png"
    cv2.imwrite(str(save_path), mosaic)
    print(f"Generated {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DnCNN comparison visualization samples.")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR.parent / "data" / "dataset_color"))
    parser.add_argument(
        "--weight_path",
        type=str,
        default=str(REPO_ROOT / "BE" / "pipeline" / "denoiser" / "weights" / "best_model.pt"),
    )
    parser.add_argument("--output_dir", type=str, default=str(BASE_DIR / "results" / "dncnn_samples"))
    parser.add_argument("--sample", type=str, default="0005.png")
    parser.add_argument("--zoom_sample", type=str, default="0002.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(device, Path(args.weight_path))
    noisy_dir = Path(args.data_dir) / "test" / "noisy"

    process_and_save(model, device, noisy_dir, output_dir, args.sample, is_zoom=False)
    process_and_save(model, device, noisy_dir, output_dir, args.zoom_sample, is_zoom=True)


if __name__ == "__main__":
    main()
