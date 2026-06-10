import argparse
import os
from pathlib import Path
import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from model import NAFNet

BASE_DIR = Path(__file__).resolve().parent


def load_state_dict(weight_path, device):
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # img shape is H, W, C in BGR
    # ssim expects data_range=255 and channel_axis=2
    return ssim(img1, img2, data_range=255, channel_axis=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate NAFNet denoising results.")
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR.parent / "data" / "dataset_color"))
    parser.add_argument(
        "--weight_path",
        type=str,
        default=str(BASE_DIR / "checkpoints_nafnet" / "best_nafnet_model_ratio40.pt"),
    )
    parser.add_argument("--output_dir", type=str, default=str(BASE_DIR / "results" / "benchmark"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load NAFNet
    model = NAFNet(in_channels=3, out_channels=3, width=32, enc_blk_nums=[1, 1, 1, 14], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    weight_path = args.weight_path
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} not found.")
        return
    model.load_state_dict(load_state_dict(weight_path, device))
    model.to(device)
    model.eval()

    noisy_dir = os.path.join(args.data_dir, "test", "noisy")
    clean_dir = os.path.join(args.data_dir, "test", "clean")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
    
    total_noisy_psnr, total_denoised_psnr = 0.0, 0.0
    total_noisy_ssim, total_denoised_ssim = 0.0, 0.0
    count = 0

    for fname in files:
        noisy_path = os.path.join(noisy_dir, fname)
        clean_path = os.path.join(clean_dir, fname)
        if not os.path.exists(clean_path): continue

        noisy_img = cv2.imread(noisy_path)
        clean_img = cv2.imread(clean_path)

        rgb_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        tensor_noisy = torch.from_numpy(rgb_noisy.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            tensor_denoised = model(tensor_noisy)
            tensor_denoised = torch.clamp(tensor_denoised, 0, 1)
        
        rgb_denoised = (tensor_denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        denoised_img = cv2.cvtColor(rgb_denoised, cv2.COLOR_RGB2BGR)

        n_psnr = calculate_psnr(noisy_img.astype(np.float32), clean_img.astype(np.float32))
        d_psnr = calculate_psnr(denoised_img.astype(np.float32), clean_img.astype(np.float32))
        n_ssim = calculate_ssim(noisy_img, clean_img)
        d_ssim = calculate_ssim(denoised_img, clean_img)

        total_noisy_psnr += n_psnr
        total_denoised_psnr += d_psnr
        total_noisy_ssim += n_ssim
        total_denoised_ssim += d_ssim
        count += 1

        # Extract "Process Visualization" for sample 0005.png
        if fname == "0005.png":
            # The residual predicted by the network is (Output - Input)
            # Since our network does x = x + inp inside, we can just compute residual = denoised - noisy
            # Wait, directly subtracting arrays might underflow, use float32
            res_img = denoised_img.astype(np.float32) - noisy_img.astype(np.float32)
            # Normalize residual for visualization: 0 is gray (128), negative is dark, positive is bright
            res_vis = np.clip((res_img * 2.0) + 128, 0, 255).astype(np.uint8)
            
            # Create a 1x3 process mosaic: Input -> Residual (What it removed/added) -> Output
            h, w = noisy_img.shape[:2]
            process_mosaic = np.zeros((h, w*3, 3), dtype=np.uint8)
            process_mosaic[:, :w] = noisy_img
            process_mosaic[:, w:w*2] = res_vis
            process_mosaic[:, w*2:] = denoised_img
            
            cv2.putText(process_mosaic, "Input (Noisy)", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(process_mosaic, "Predicted Residual", (w + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(process_mosaic, "Output (Denoised)", (w*2 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            cv2.imwrite(os.path.join(output_dir, "denoise_process_0005.png"), process_mosaic)

    print(f"\nBenchmark over {count} pairs:")
    print(f"Avg Noisy PSNR:    {total_noisy_psnr/count:.2f} dB")
    print(f"Avg Denoised PSNR: {total_denoised_psnr/count:.2f} dB")
    print(f"Avg Noisy SSIM:    {total_noisy_ssim/count:.4f}")
    print(f"Avg Denoised SSIM: {total_denoised_ssim/count:.4f}")

if __name__ == "__main__":
    main()
