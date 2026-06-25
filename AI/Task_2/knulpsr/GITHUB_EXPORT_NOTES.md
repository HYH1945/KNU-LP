# GitHub Export Notes

This folder is the split export for the `knulpsr` module, with RGDiffSR used as the baseline architecture.

Large and local artifacts are intentionally excluded: checkpoints, recognizer weights, cached bytecode, logs, generated evaluation outputs, and local datasets.

Public config files are intentionally minimized:

- `configs/latent-diffusion/sr_best.yaml`: SR training config
- `configs/latent-diffusion/sr_test.yaml`: SR test/inference config
- `configs/autoencoder/vqgan_2x_plate_allhr.yaml`: VQGAN first-stage training config

Path placeholders used in configs/code:

- `<TRAIN_DATA_ROOT>`, `<VAL_DATA_ROOT>`, `<TEST_DATA_ROOT>`: local dataset roots for each split
- `<DATA_ROOT>`: generic local dataset root used by older placeholders
- `<MODEL_CHECKPOINT_PATH>`: trained knulpsr/VQGAN checkpoint path
- `<GPLPR_ROOT>` and `<GPLPR_CHECKPOINT_PATH>`: optional GPLPR/KNU-LP-OCR repo/checkpoint, expected to live outside this split export
- `<TEXTZOOM_ROOT>`, `<TEXTZOOM_TRAIN_ROOT>`, `<TEXTZOOM_TEST_ROOT>`, `<OCR_DATA_ROOT>`, `<SR_DATA_ROOT>`, `<REAL_SR_DATA_ROOT>`, `<SYNTH_TEXT_ROOT>`, `<GT_IMAGE_ROOT>`, `<RESTORED_IMAGE_ROOT>`: dataset roots used by legacy/debug data utilities
- `<SYNTH_TEXT_ANNOTATION_PATH>`, `<ICDAR_ANNOTATION_PATH>`, `<ICDAR13_TRAIN_LMDB_ROOT>`, `<ICDAR13_TEST_LMDB_ROOT>`, `<ICDAR15_TRAIN_LMDB_ROOT>`, `<ICDAR15_TEST_LMDB_ROOT>`, `<SVTP_DATA_ROOT>`, `<SVTP_LMDB_ROOT>`: optional dataset-conversion placeholders
- `<OCCLUDED_DATA_ROOT>`, `<OCCLUDED_PAIR_ROOT>`, `<OCCLUDED_TRAIN_ROOT>`, `<OCCLUDED_TEST_ROOT>`, `<OCCLUDED_SOURCE_ROOT_A>`, `<OCCLUDED_SOURCE_ROOT_B>`, `<OCR_BENCHMARK_ROOT>`: optional occlusion/debug dataset roots
- `<LMDB_ROOT>`, `<LMDB_OUTPUT_ROOT>`, `<VISIONLAN_DICT_PATH>`, `<VISIONLAN_CHECKPOINT_PATH>`, and `<FONT_PATH>`: optional legacy auxiliary resources

PARSeq TPG is defined in `ldm/modules/encoders/tp_generator.py` as `PARSeqTPG`. The active configs load an existing PARSeq checkpoint such as `checkpoint/parseq.pt` and usually set `freeze_backbone: true`, so PARSeq is used as a pretrained/frozen recognizer prior unless a config explicitly changes that.
