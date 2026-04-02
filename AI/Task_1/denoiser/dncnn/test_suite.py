import sys
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 상위 디렉토리 import 강제
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dncnn.evaluate import PlateDenoiseDataset, DnCNN, evaluate_model, Config, load_checkpoint, print_eval_results

# Identity Mapping (깨끗한 원본) 테스트를 위한 데이터로더 오버라이딩
class CleanDataset(PlateDenoiseDataset):
    def __init__(self, split_root, image_size):
        super().__init__(split_root, image_size)
        self.noisy_dir = self.clean_dir

def measure_latency(model, loader, device, num_runs=100):
    model.eval()
    
    # 1. Warm-up
    batch = next(iter(loader))
    x = batch["input"].to(device)
    for _ in range(10):
        _ = model(x)
        
    start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    
    if device == "cuda":
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.time()
        
    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["input"].to(device)
            _ = model(imgs)
            count += imgs.size(0)
            if count >= num_runs:
                break
                
    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event) / 1000.0  # 초 단위 변환
    else:
        total_time = time.time() - start_time
        
    fps = count / total_time
    latency_ms = (total_time / count) * 1000
    return latency_ms, fps

def save_visual_samples(model, loader, device, save_path="results_samples.png", num_samples=3, title="Model Output"):
    model.eval()
    batch = next(iter(loader))
    x = batch["input"].to(device)
    y = batch["target"].to(device)
    filenames = batch["filename"]
    
    with torch.no_grad():
        pred = model(x)
        
    num_samples = min(num_samples, x.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(num_samples):
        inp_img = x[i].cpu().squeeze(0).numpy()
        pred_img = pred[i].cpu().squeeze(0).numpy()
        target_img = y[i].cpu().squeeze(0).numpy()
        
        axes[i, 0].imshow(inp_img, cmap="gray")
        axes[i, 0].set_title(f"Noisy Input\n{filenames[i]}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(pred_img, cmap="gray")
        axes[i, 1].set_title("Model Output")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(target_img, cmap="gray")
        axes[i, 2].set_title("Target (GT)")
        axes[i, 2].axis("off")
        
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 결과물 이미지 저장 완료: {save_path}")

def main():
    cfg = Config()
    
    print("="*60)
    print(" [ 1. 정량적 평가 (Quantitative Evaluation) ]")
    print("="*60)
    
    # 일반 데이터 평가
    test_set = PlateDenoiseDataset(cfg.test_root, cfg.image_size)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)
    
    model = DnCNN(in_channels=1, depth=17, features=64).to(cfg.device)
    model = load_checkpoint(model, cfg.checkpoint_path, cfg.device)
    
    print("\n---> Noisy Test (거친 노이즈가 있는 일반 테스트셋)")
    norm_results = evaluate_model(model, test_loader, cfg.device, cfg.pixel_acc_threshold)
    print_eval_results(norm_results, title="Standard Noisy Test")
    
    # Identity 데이터 평가 (원본 그대로 전달)
    print("\n---> Identity Test (깨끗한 원본이 들어왔을 때 훼손하지 않는가?)")
    id_set = CleanDataset(cfg.test_root, cfg.image_size)
    id_loader = DataLoader(id_set, batch_size=cfg.batch_size, shuffle=False)
    
    id_results = evaluate_model(model, id_loader, cfg.device, cfg.pixel_acc_threshold)
    print_eval_results(id_results, title="Identity Mapping Test")
    
    print("\n" + "="*60)
    print(" [ 2. 정성적 평가 / 시각화 (Qualitative Evaluation) ]")
    print("="*60)
    
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    
    save_visual_samples(model, test_loader, cfg.device, save_path=str(save_dir / "standard_noisy_samples.png"), title="Standard Denoising (Noisy -> Clean)")
    save_visual_samples(model, id_loader, cfg.device, save_path=str(save_dir / "identity_mapping_samples.png"), title="Identity Mapping (Clean -> Clean)")
    
    print("\n" + "="*60)
    print(" [ 3. 실시간 처리 속도 측정 (Real-time Latency) ]")
    print("="*60)
    
    # 실시간 처리 환경을 위해 batch_size = 1 로 측정
    single_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    latency, fps = measure_latency(model, single_loader, cfg.device, num_runs=100)
    
    print(f"Device        : {cfg.device}")
    print(f"Batch Size    : 1")
    print(f"Latency / img : {latency:.2f} ms")
    print(f"FPS           : {fps:.2f} frames/sec")
    print("\n모든 테스트가 완료되었습니다.")

if __name__ == "__main__":
    main()
