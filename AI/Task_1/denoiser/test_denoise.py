import os
import torch
import cv2
import numpy as np
from models.dncnn import DnCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 모델 로드
    model = DnCNN(in_channels=3)
    weights_path = "checkpoints/best_color_model.pt"
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. 결과 저장 경로
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 테스트 데이터셋 경로
    test_noisy_dir = "data/dataset_color/test/noisy"
    test_clean_dir = "data/dataset_color/test/clean"
    
    # 테스트할 샘플 목록 (번호판 이미지 5개 추출)
    samples = sorted(os.listdir(test_noisy_dir))[:5]
    
    print(f"Processing {len(samples)} samples...")
    for idx, fname in enumerate(samples):
        noisy_path = os.path.join(test_noisy_dir, fname)
        clean_path = os.path.join(test_clean_dir, fname)
        
        # 이미지 로드
        img_noisy = cv2.imread(noisy_path)
        img_clean = cv2.imread(clean_path)
        
        if img_noisy is None or img_clean is None:
            continue
            
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB)
        
        # 전처리
        normalized = img_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 추론
        with torch.no_grad():
            output = model(tensor)
            
        # 후처리
        output_np = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
        img_denoised = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        # 시각화를 위해 hconcat: [Noisy | Denoised | Clean]
        scale = 3
        h, w = img_noisy.shape[:2]
        new_size = (w * scale, h * scale)
        
        r_noisy = cv2.resize(img_noisy, new_size, interpolation=cv2.INTER_NEAREST)
        r_denoised = cv2.resize(img_denoised, new_size, interpolation=cv2.INTER_NEAREST)
        r_clean = cv2.resize(img_clean, new_size, interpolation=cv2.INTER_NEAREST)
        
        # 각 이미지 위에 텍스트 라벨 쓰기
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(r_noisy, "Noisy (Input)", (10, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(r_denoised, "Denoised (Output)", (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(r_clean, "Clean (GT)", (10, 25), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        # 이미지 결합
        comparison = cv2.hconcat([r_noisy, r_denoised, r_clean])
        
        # 저장
        save_path = os.path.join(output_dir, f"compare_{fname}")
        cv2.imwrite(save_path, comparison)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
