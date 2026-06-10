import os
import torch
import cv2
import numpy as np
from model import NAFNet

device = torch.device('cuda')
model = NAFNet(in_channels=3, out_channels=3, width=32, enc_blk_nums=[1, 1, 1, 14], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
model.load_state_dict(torch.load('checkpoints_nafnet/best_nafnet_model.pt', map_location=device, weights_only=True))
model.to(device)
model.eval()

samples = ['0001.png', '0002.png', '0004.png']
out_dir = 'C:/Users/jech0/.gemini/antigravity/brain/1b03cfae-ed43-4bd6-9bc3-4932bdeb88ea/'
noisy_dir = 'c:/Users/jech0/Desktop/projects/KNU-LP/dataset_denoising_color/test/noisy/'

for fname in samples:
    noisy_img = cv2.imread(os.path.join(noisy_dir, fname))
    rgb_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    tensor_noisy = torch.from_numpy(rgb_noisy.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        tensor_denoised = torch.clamp(model(tensor_noisy), 0, 1)
        
    rgb_denoised = (tensor_denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    denoised_img = cv2.cvtColor(rgb_denoised, cv2.COLOR_RGB2BGR)
    
    res_img = denoised_img.astype(np.float32) - noisy_img.astype(np.float32)
    res_vis = np.clip((res_img * 2.0) + 128, 0, 255).astype(np.uint8)
    
    h, w = noisy_img.shape[:2]
    process_mosaic = np.zeros((h, w*3, 3), dtype=np.uint8)
    process_mosaic[:, :w] = noisy_img
    process_mosaic[:, w:w*2] = res_vis
    process_mosaic[:, w*2:] = denoised_img
    
    cv2.putText(process_mosaic, 'Input (Noisy)', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(process_mosaic, 'Predicted Residual', (w + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(process_mosaic, 'Output (Denoised)', (w*2 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    cv2.imwrite(os.path.join(out_dir, f'denoise_process_{fname}'), process_mosaic)
    print(f"Generated denoise_process_{fname}")
