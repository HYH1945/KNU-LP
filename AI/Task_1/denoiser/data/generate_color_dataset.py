"""
컬러(RGB) 복합 열화(Complex Degradation) 디노이징 데이터셋 생성 스크립트
- 기존 흑백 데이터셋(data/dataset)은 절대 건드리지 않음
- 새 폴더: data/dataset_color/{train,val,test}/{clean,noisy}
- 원본 컬러 이미지 → clean
- 복합 노이즈(Gaussian, Salt&Pepper, JPEG, Motion Blur, Defocus Blur, Brightness, Down-Up) → noisy
- 분할 비율: train 80% / val 10% / test 10%
"""

import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np

# ── 설정 ──────────────────────────────────────────────
SOURCE_DIR = Path(__file__).resolve().parents[2] / "train"  # AI/Task_1/train
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_color"

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
# ─────────────────────────────────────────────────────

# 1. 가우시안 노이즈 (센서 노이즈)
def add_gaussian_noise(image: np.ndarray, sigma_range=(10, 45)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float64)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)

# 2. 솔트&페퍼 노이즈 (픽셀 손상)
def add_salt_pepper_noise(image: np.ndarray, prob_range=(0.002, 0.015)) -> np.ndarray:
    prob = random.uniform(*prob_range)
    noisy = image.copy()
    # Salt (white pixels)
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper (black pixels)
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

# 3. JPEG 압축 손실
def add_jpeg_compression(image: np.ndarray, quality_range=(30, 80)) -> np.ndarray:
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_COLOR)

# 4. 이동 블러 (Motion Blur)
def add_motion_blur(image: np.ndarray, size_range=(3, 7)) -> np.ndarray:
    size = random.choice([i for i in range(size_range[0], size_range[1] + 1) if i % 2 == 1])
    kernel = np.zeros((size, size))
    direction = random.choice(['h', 'v', 'd1', 'd2'])
    if direction == 'h':
        kernel[int((size-1)/2), :] = 1
    elif direction == 'v':
        kernel[:, int((size-1)/2)] = 1
    elif direction == 'd1':
        for i in range(size):
            kernel[i, i] = 1
    else:
        for i in range(size):
            kernel[i, size - i - 1] = 1
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

# 5. 초점 흐림 (Defocus Blur)
def add_defocus_blur(image: np.ndarray, kernel_range=(3, 7)) -> np.ndarray:
    ksize = random.choice([i for i in range(kernel_range[0], kernel_range[1] + 1) if i % 2 == 1])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# 6. 조명 및 대비 변화
def adjust_brightness_contrast(image: np.ndarray, brightness_range=(-20, 20), contrast_range=(0.8, 1.2)) -> np.ndarray:
    alpha = random.uniform(*contrast_range)
    beta = random.uniform(*brightness_range)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 7. 저해상도 샘플링 후 복원 (Down-Up Sampling)
def add_downup_sampling(image: np.ndarray, scale_range=(0.4, 0.8)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    nh, nw = max(int(h * scale), 4), max(int(w * scale), 4)
    # Downsample
    down = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    # Upsample back
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

# 종합 복합 열화 파이프라인
def apply_complex_noise(image: np.ndarray) -> np.ndarray:
    noisy = image.copy()
    
    # 1. Blur (40% prob)
    if random.random() < 0.4:
        if random.random() < 0.5:
            noisy = add_defocus_blur(noisy)
        else:
            noisy = add_motion_blur(noisy)
            
    # 2. Down-Up Sampling (30% prob)
    if random.random() < 0.3:
        noisy = add_downup_sampling(noisy)
        
    # 3. Brightness & Contrast (40% prob)
    if random.random() < 0.4:
        noisy = adjust_brightness_contrast(noisy)
        
    # 4. Gaussian Noise (70% prob)
    if random.random() < 0.7:
        noisy = add_gaussian_noise(noisy)
        
    # 5. Salt & Pepper Noise (10% prob)
    if random.random() < 0.1:
        noisy = add_salt_pepper_noise(noisy)
        
    # 6. JPEG Compression (50% prob)
    if random.random() < 0.5:
        noisy = add_jpeg_compression(noisy)
        
    return noisy


def main():
    # 소스 이미지 수집
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    source_files = sorted(
        [f for f in SOURCE_DIR.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    )
    print(f"[INFO] Source image count: {len(source_files)} ({SOURCE_DIR})")

    if not source_files:
        raise FileNotFoundError(f"No source images found in: {SOURCE_DIR}")

    # 셔플 및 분할
    random.seed(SEED)
    indices = list(range(len(source_files)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * SPLITS["train"])
    n_val = int(n_total * SPLITS["val"])

    split_map = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    # 출력 디렉토리 생성
    for split in SPLITS:
        (OUTPUT_DIR / split / "clean").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "noisy").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    print(f"[INFO] Splits: train={len(split_map['train'])} / val={len(split_map['val'])} / test={len(split_map['test'])}")

    # 데이터셋 생성
    global_idx = 0

    for split, idxs in split_map.items():
        for i, src_idx in enumerate(idxs):
            src_path = source_files[src_idx]
            img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Failed to read, skipping: {src_path}")
                continue

            fname = f"{i:04d}.png"

            # clean: 원본 컬러
            clean_path = OUTPUT_DIR / split / "clean" / fname
            cv2.imwrite(str(clean_path), img)

            # noisy: 복합 노이즈 추가
            noisy_img = apply_complex_noise(img)
            noisy_path = OUTPUT_DIR / split / "noisy" / fname
            cv2.imwrite(str(noisy_path), noisy_img)

            global_idx += 1

        print(f"  [OK] {split}: {len(idxs)} done")

    print(f"\n[DONE] Color dataset generation complete! Total {global_idx} pairs generated.")
    print(f"  Path: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
