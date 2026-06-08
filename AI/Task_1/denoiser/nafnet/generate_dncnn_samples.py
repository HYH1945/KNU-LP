import os
import cv2
import numpy as np
import torch
import sys

# DnCNN requires being imported from the project path if we use its code
sys.path.append('c:/Users/jech0/Desktop/projects/KNU-LP/BE')
from pipeline.denoiser.model import DnCNN

device = torch.device('cuda')
model = DnCNN(in_channels=3, depth=17, features=64)
# Load Color DnCNN weights
model.load_state_dict(torch.load('c:/Users/jech0/Desktop/projects/KNU-LP/BE/pipeline/denoiser/weights/best_model.pt', map_location=device, weights_only=True))
model.to(device)
model.eval()

out_dir = 'C:/Users/jech0/.gemini/antigravity/brain/1b03cfae-ed43-4bd6-9bc3-4932bdeb88ea/'

def process_and_save(fname, is_zoom=False):
    noisy_path = f'c:/Users/jech0/Desktop/projects/KNU-LP/dataset_denoising_color/test/noisy/{fname}'
    noisy_img = cv2.imread(noisy_path)
    rgb_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
    tensor_noisy = torch.from_numpy(rgb_noisy.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        tensor_denoised = model(tensor_noisy)
        
    rgb_denoised = (tensor_denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    denoised_img = cv2.cvtColor(rgb_denoised, cv2.COLOR_RGB2BGR)
    
    if not is_zoom:
        # Process visualization
        res_img = noisy_img.astype(np.float32) - denoised_img.astype(np.float32)
        # We invert the sign so it looks similar to the NAFNet visualization
        res_vis = np.clip((res_img * 2.0) + 128, 0, 255).astype(np.uint8)
        
        h, w = noisy_img.shape[:2]
        mosaic = np.zeros((h, w*3, 3), dtype=np.uint8)
        mosaic[:, :w] = noisy_img
        mosaic[:, w:w*2] = res_vis
        mosaic[:, w*2:] = denoised_img
        
        cv2.putText(mosaic, 'Input (Noisy)', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(mosaic, 'Predicted Residual', (w + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(mosaic, 'Output (DnCNN)', (w*2 + 10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        cv2.imwrite(os.path.join(out_dir, f'dncnn_process_{fname}'), mosaic)
        print(f"Saved dncnn_process_{fname}")
    else:
        # Zoomed-in visualization
        crop_n = noisy_img[10:30, 45:65]
        crop_d = denoised_img[10:30, 45:65]

        crop_n = cv2.resize(crop_n, (200, 200), interpolation=cv2.INTER_NEAREST)
        crop_d = cv2.resize(crop_d, (200, 200), interpolation=cv2.INTER_NEAREST)

        h, w = crop_n.shape[:2]
        mosaic = np.zeros((h, w*2, 3), dtype=np.uint8)
        mosaic[:, :w] = crop_n
        mosaic[:, w:] = crop_d

        cv2.putText(mosaic, 'Noisy (Zoomed)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(mosaic, 'DnCNN (Zoomed)', (w+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cv2.imwrite(os.path.join(out_dir, 'dncnn_zoom_proof.png'), mosaic)
        print("Saved dncnn_zoom_proof.png")

# Process 0005 for residual
process_and_save('0005.png', is_zoom=False)
# Process 0002 for zoom
process_and_save('0002.png', is_zoom=True)
