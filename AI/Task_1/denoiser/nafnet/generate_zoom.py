import os
import cv2
import numpy as np
import torch
from model import NAFNet

device = torch.device('cuda')
model = NAFNet(in_channels=3, out_channels=3, width=32, enc_blk_nums=[1, 1, 1, 14], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
model.load_state_dict(torch.load('checkpoints_nafnet/best_nafnet_model.pt', map_location=device, weights_only=True))
model.to(device)
model.eval()

noisy = cv2.imread('c:/Users/jech0/Desktop/projects/KNU-LP/dataset_denoising_color/test/noisy/0002.png')
rgb_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
tensor_noisy = torch.from_numpy(rgb_noisy.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)

with torch.no_grad():
    tensor_denoised = torch.clamp(model(tensor_noisy), 0, 1)

rgb_denoised = (tensor_denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
denoised_img = cv2.cvtColor(rgb_denoised, cv2.COLOR_RGB2BGR)

# Crop a small region that has clear noise
crop_n = noisy[10:30, 45:65]
crop_d = denoised_img[10:30, 45:65]

# Scale up using NEAREST interpolation to make pixels visible
crop_n = cv2.resize(crop_n, (200, 200), interpolation=cv2.INTER_NEAREST)
crop_d = cv2.resize(crop_d, (200, 200), interpolation=cv2.INTER_NEAREST)

h, w = crop_n.shape[:2]
mosaic = np.zeros((h, w*2, 3), dtype=np.uint8)
mosaic[:, :w] = crop_n
mosaic[:, w:] = crop_d

cv2.putText(mosaic, 'Noisy (Zoomed)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(mosaic, 'Denoised (Zoomed)', (w+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

out_dir = 'C:/Users/jech0/.gemini/antigravity/brain/1b03cfae-ed43-4bd6-9bc3-4932bdeb88ea/'
cv2.imwrite(os.path.join(out_dir, 'zoom_proof.png'), mosaic)
print("Saved zoom_proof.png")
