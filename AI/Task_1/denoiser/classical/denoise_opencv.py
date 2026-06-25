import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to access utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.metrics import mae_np as mae, mse_np as mse, psnr_np as psnr, ssim_np as ssim, pixel_accuracy_np as pixel_accuracy
from utils.visualize import show_samples_for_method, show_comparison_across_methods

# 설정
class Config:
    test_root = "../data/dataset/test"
    image_size = (48, 128)
    sample_count = 3            # 시각화할 샘플 수
    pixel_acc_threshold = 0.05  # 0~1 정규화 기준

# 이미지 로딩 유틸
def load_gray_image(path, image_size=(48, 128), normalize=True):

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"이미지 읽기 실패: {path}")

    h, w = image_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    if normalize:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    return img


def get_paired_filenames(split_root):

    clean_dir = Path(split_root) / "clean"
    noisy_dir = Path(split_root) / "noisy"

    if not clean_dir.exists():
        raise FileNotFoundError(f"clean 폴더 없음: {clean_dir}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"noisy 폴더 없음: {noisy_dir}")

    clean_files = sorted([f.name for f in clean_dir.iterdir() if f.is_file()])
    noisy_files = set([f.name for f in noisy_dir.iterdir() if f.is_file()])

    paired = [f for f in clean_files if f in noisy_files]

    if len(paired) == 0:
        raise ValueError("짝이 맞는 이미지가 없습니다.")

    return paired


# 고전적 디노이징 함수들
def apply_bilateral(img_uint8, d=5, sigma_color=50, sigma_space=50):
    return cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)

def apply_nlm(img_uint8, h=10, template_window_size=7, search_window_size=21):
    return cv2.fastNlMeansDenoising(
        src=img_uint8,
        h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )

def apply_median(img_uint8, ksize=3):
    return cv2.medianBlur(img_uint8, ksize)





# 전체 평가
def evaluate_method(split_root, method_name, image_size=(48, 128), threshold=0.05):

    filenames = get_paired_filenames(split_root)

    clean_dir = Path(split_root) / "clean"
    noisy_dir = Path(split_root) / "noisy"

    metric_list = {
        "mae": [],
        "mse": [],
        "psnr": [],
        "ssim": [],
        "acc": [],
    }

    sample_results = []

    for fname in filenames:
        gt = load_gray_image(clean_dir / fname, image_size=image_size, normalize=False)
        inp = load_gray_image(noisy_dir / fname, image_size=image_size, normalize=False)

        # 고전 방식 적용
        if method_name == "bilateral":
            out = apply_bilateral(inp, d=5, sigma_color=50, sigma_space=50)
        elif method_name == "nlm":
            out = apply_nlm(inp, h=10, template_window_size=7, search_window_size=21)
        elif method_name == "median":
            out = apply_median(inp, ksize=3)
        else:
            raise ValueError(f"지원하지 않는 method: {method_name}")

        # 0~1 정규화 후 평가
        gt_f = gt.astype(np.float32) / 255.0
        inp_f = inp.astype(np.float32) / 255.0
        out_f = out.astype(np.float32) / 255.0

        metric_list["mae"].append(mae(out_f, gt_f))
        metric_list["mse"].append(mse(out_f, gt_f))
        metric_list["psnr"].append(psnr(out_f, gt_f))
        metric_list["ssim"].append(ssim(out_f, gt_f))
        metric_list["acc"].append(pixel_accuracy(out_f, gt_f, threshold=threshold))

        sample_results.append({
            "filename": fname,
            "input": inp_f,
            "output": out_f,
            "target": gt_f,
        })

    summary = {k: float(np.mean(v)) for k, v in metric_list.items()}
    return summary, sample_results


# 결과 출력
def print_summary_table(results_dict):
    # 방법별 평균 성능 출력
    print("\n================ Classical Denoising Summary ================\n")
    header = f"{'Method':<12} {'MAE':>10} {'MSE':>10} {'PSNR':>10} {'SSIM':>10} {'ACC':>10}"
    print(header)
    print("-" * len(header))

    for method_name, summary in results_dict.items():
        print(
            f"{method_name:<12} "
            f"{summary['mae']:>10.6f} "
            f"{summary['mse']:>10.6f} "
            f"{summary['psnr']:>10.4f} "
            f"{summary['ssim']:>10.4f} "
            f"{summary['acc']:>10.4f}"
        )

    print()






# 메인 실행
def main():
    cfg = Config()

    methods = ["bilateral", "nlm", "median"]
    results_dict = {}
    method_samples_dict = {}

    for method in methods:
        summary, sample_results = evaluate_method(
            split_root=cfg.test_root,
            method_name=method,
            image_size=cfg.image_size,
            threshold=cfg.pixel_acc_threshold
        )
        results_dict[method] = summary
        method_samples_dict[method] = sample_results

    # 전체 통계 출력
    print_summary_table(results_dict)

    # 방법별 샘플 출력
    for method in methods:
        print(f"[INFO] show samples for: {method}")
        show_samples_for_method(
            method_name=method,
            sample_results=method_samples_dict[method],
            sample_count=cfg.sample_count
        )

    # 한 샘플에 대해 여러 방법 비교
    print("[INFO] compare methods on one sample")
    show_comparison_across_methods(
        method_samples_dict=method_samples_dict,
        sample_index=0
    )


if __name__ == "__main__":
    main()