# Backend Weights

Place model weights in these BE-local folders:

```text
BE/pipeline/plate_cropper/weights/best.pt
BE/pipeline/denoiser/weights/best_model.pt
BE/pipeline/superresolution/weights/SR.ckpt
BE/pipeline/superresolution/weights/VQGAN.ckpt
BE/pipeline/superresolution/weights/parseq.pt
BE/pipeline/ocr/weights/best_model.pth
```

Usage:

- `best.pt`: YOLO license plate detector.
- `best_model.pt`: DnCNN denoiser.
- `SR.ckpt`: RGDiffSR 5-frame diffusion checkpoint.
- `VQGAN.ckpt`: RGDiffSR first-stage VQGAN checkpoint from the 2026-05-08T10-23-54 SR test setup.
- `parseq.pt`: PARSeq checkpoint used by the SR conditioning module.
- `best_model.pth`: GP-LPR OCR checkpoint.

Code vendors:

- SR code is vendored under `BE/pipeline/superresolution/rgdiffsr_vendor`.
- GP-LPR OCR model code is vendored under `BE/pipeline/ocr/gplpr_vendor`.

Runtime settings live in `BE/configs/settings.yaml`.
