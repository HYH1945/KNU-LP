import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


# 개별 노이즈 함수
def add_gaussian_noise(img, sigma=10):
    """
    가우시안 노이즈
    - 저조도 CCTV, 센서 노이즈 상황을 단순 모사
    - sigma가 클수록 더 강한 노이즈
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.002):
    """
    salt & pepper 노이즈
    - 일부 픽셀을 검정/흰색으로 강하게 훼손
    - 현실성은 낮지만 심한 손상 보조용으로 사용
    """
    out = img.copy()
    num_pixels = int(amount * img.shape[0] * img.shape[1])

    coords = (
        np.random.randint(0, img.shape[0], num_pixels),
        np.random.randint(0, img.shape[1], num_pixels)
    )
    out[coords] = 255

    coords = (
        np.random.randint(0, img.shape[0], num_pixels),
        np.random.randint(0, img.shape[1], num_pixels)
    )
    out[coords] = 0

    return out


def add_jpeg_compression(img, quality=40):
    """
    JPEG 압축 열화
    - CCTV 저장/전송 과정의 손실 압축을 모사
    - quality가 낮을수록 더 심하게 깨짐
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, encimg = cv2.imencode(".jpg", img, encode_param)
    if not success:
        return img.copy()

    decoded = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        return img.copy()
    return decoded


def add_motion_blur(img, kernel_size=5):
    """
    모션 블러
    - 차량 이동, 흔들림 상황 모사
    - kernel_size가 클수록 더 번짐
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def add_defocus_blur(img, k=3):
    """
    초점 흐림
    - 카메라 초점 불량, 먼 거리 촬영 상황 모사
    - k가 클수록 더 흐려짐
    """
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """
    밝기 / 대비 변화
    - alpha < 1 : 대비 감소
    - beta < 0  : 어두워짐
    """
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def down_up_sample(img, scale=0.5):
    """
    다운샘플 후 업샘플
    - 원거리 촬영 / 저해상도 CCTV 느낌 생성
    - scale이 작을수록 더 심하게 열화됨
    """
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored


# 복합 열화
def degrade_image(img, strong=True):
    """
    개별 노이즈 함수를 복합적으로 적용
    """
    out = img.copy()

    if strong:
        if random.random() < 0.9:
            out = add_defocus_blur(out, k=random.choice([3, 5, 7]))

        if random.random() < 0.7:
            out = add_motion_blur(out, kernel_size=random.choice([5, 7, 9]))

        if random.random() < 0.95:
            out = down_up_sample(out, scale=random.uniform(0.25, 0.6))

        if random.random() < 0.9:
            out = add_jpeg_compression(out, quality=random.randint(10, 35))

        if random.random() < 0.9:
            out = add_gaussian_noise(out, sigma=random.uniform(12, 30))

        if random.random() < 0.3:
            out = add_salt_pepper_noise(out, amount=0.003)

        if random.random() < 0.85:
            out = adjust_brightness_contrast(
                out,
                alpha=random.uniform(0.5, 0.9),
                beta=random.uniform(-50, -10)
            )
    else:
        if random.random() < 0.7:
            out = add_defocus_blur(out, k=random.choice([3, 5]))

        if random.random() < 0.5:
            out = add_motion_blur(out, kernel_size=random.choice([3, 5, 7]))

        if random.random() < 0.8:
            out = down_up_sample(out, scale=random.uniform(0.4, 0.8))

        if random.random() < 0.8:
            out = add_jpeg_compression(out, quality=random.randint(20, 60))

        if random.random() < 0.8:
            out = add_gaussian_noise(out, sigma=random.uniform(5, 20))

        if random.random() < 0.2:
            out = add_salt_pepper_noise(out, amount=0.0015)

        if random.random() < 0.7:
            out = adjust_brightness_contrast(
                out,
                alpha=random.uniform(0.7, 1.1),
                beta=random.uniform(-30, 10)
            )

    return out


# 데이터셋 생성 (train / val / test 분할)
def create_dataset(input_dir, output_root, preview_count=5, strong=True):
    file_list = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ])

    random.shuffle(file_list)

    total = len(file_list)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train": file_list[:train_end],
        "val": file_list[train_end:val_end],
        "test": file_list[val_end:]
    }

    preview_samples = []

    for split_name, files in splits.items():
        clean_dir = os.path.join(output_root, split_name, "clean")
        noisy_dir = os.path.join(output_root, split_name, "noisy")

        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        for fname in files:
            input_path = os.path.join(input_dir, fname)

            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            noisy = degrade_image(img, strong=strong)

            cv2.imwrite(os.path.join(clean_dir, fname), img)
            cv2.imwrite(os.path.join(noisy_dir, fname), noisy)

            if len(preview_samples) < preview_count and split_name == "train":
                preview_samples.append((fname, img, noisy))

    print("[INFO] Dataset generation completed.")
    print(f"[INFO] Total images: {total}")

    show_preview_samples(preview_samples)


# 시각화
def show_preview_samples(sample_pairs):
    if len(sample_pairs) == 0:
        return

    rows = len(sample_pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 3 * rows))

    if rows == 1:
        axes = np.array([axes])

    for i, (fname, clean_img, noisy_img) in enumerate(sample_pairs):
        axes[i, 0].imshow(clean_img, cmap="gray")
        axes[i, 0].set_title(f"Clean\n{fname}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(noisy_img, cmap="gray")
        axes[i, 1].set_title(f"Noisy\n{fname}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# main
if __name__ == "__main__":
    INPUT_DIR = "plates"
    OUTPUT_ROOT = "dataset"

    create_dataset(
        input_dir=INPUT_DIR,
        output_root=OUTPUT_ROOT,
        preview_count=5,
        strong=True
    )