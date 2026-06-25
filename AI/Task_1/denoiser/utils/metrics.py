import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------------------------------------
# NumPy 기반 지표 (주로 OpenCV 등 고전 알고리즘에서 활용)
# -------------------------------------------------------------
def mae_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))

def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))

def psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    mse_val = mse_np(pred, target)
    if mse_val == 0:
        return 100.0
    return float(10 * np.log10(1.0 / mse_val))

def ssim_np(pred: np.ndarray, target: np.ndarray) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = pred.mean()
    mu_y = target.mean()

    var_x = pred.var()
    var_y = target.var()
    cov_xy = ((pred - mu_x) * (target - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)

    return float(numerator / (denominator + 1e-8))

def pixel_accuracy_np(pred: np.ndarray, target: np.ndarray, threshold: float = 0.05) -> float:
    """보조 지표 - pred와 GT의 차이가 threshold 이하인 픽셀 비율"""
    diff = np.abs(pred - target)
    return float(np.mean(diff <= threshold))


# -------------------------------------------------------------
# PyTorch 기반 지표 (딥러닝 모델 학습 / 추론에서 활용)
# -------------------------------------------------------------
def psnr_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse_val = F.mse_loss(pred, target).item()
    if mse_val == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse_val)

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

def pixel_accuracy_torch(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05) -> float:
    """보조 지표 - pred와 GT의 차이가 threshold 이하인 픽셀 비율"""
    diff = torch.abs(pred - target)
    correct = (diff <= threshold).float().mean().item()
    return correct
