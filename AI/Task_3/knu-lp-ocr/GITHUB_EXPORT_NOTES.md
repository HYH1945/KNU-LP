# GitHub Export Notes

This folder is the split export for the `knu-lp-ocr` module, based on GPLPR training code.

Included:

- Training and testing entry points: `train.py`, `test_ocr.py`
- Model, dataset, loss, and training helper modules
- Minimal YAML configs: `config/train.yaml`, `config/test.yaml`
- Public dataset split examples without plate-label annotations

Excluded:

- `save/` experiment outputs
- `train_dir/` sample images and plate-label annotations
- Checkpoints and recognizer weights
- Python caches, IDE metadata, local datasets, and generated logs

Path placeholders should be updated in the YAML config files before training on another machine.
