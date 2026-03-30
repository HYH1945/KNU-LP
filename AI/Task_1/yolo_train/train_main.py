"""
train_main.py
- YOLO segmentation 학습 파이프라인의 진입점
- DatasetValidator → ModelTrainer 순서로 실행
- --resume 플래그로 중단된 학습 재개 가능
"""

import argparse
import sys
import os

# 모듈 경로 등록
sys.path.insert(0, os.path.dirname(__file__))

from modules.dataset_validator import DatasetValidator
from modules.train_config       import TrainConfig
from modules.model_trainer      import ModelTrainer


# ====================================================================== #
def parse_args() -> argparse.Namespace:
    """
    Input : sys.argv
    Output: argparse.Namespace — 파싱된 CLI 인자
    """
    parser = argparse.ArgumentParser(description="YOLO seg 번호판 학습")

    parser.add_argument("--data",      type=str, default="dataset.yaml",      help="dataset.yaml 경로")
    parser.add_argument("--weights",   type=str, default="yolov8n-seg.pt",    help="사전학습 가중치")
    parser.add_argument("--epochs",    type=int, default=100)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--device",    type=str, default="0",                 help="'0', 'cpu', '0,1'")
    parser.add_argument("--name",      type=str, default="plate_seg",         help="실험 이름")
    parser.add_argument("--project", type=str, default="runs/segment")
    parser.add_argument("--resume",    action="store_true",                   help="마지막 체크포인트에서 재개")
    parser.add_argument("--skip-val",  action="store_true",                   help="데이터셋 검증 생략")
    parser.add_argument("--config",    type=str, default=None,                help="JSON 설정 파일 경로 (있으면 CLI 인자 덮어씀)")

    return parser.parse_args()


# ====================================================================== #
def main():
    args = parse_args()

    # ── 1. 설정 로드 ────────────────────────────────────────────────── #
    if args.config:
        cfg = TrainConfig.load(args.config)
        print(f"[main] 설정 파일 로드: {args.config}")
    else:
        cfg = TrainConfig(
            dataset_yaml      = args.data,
            pretrained_weights= args.weights,
            epochs            = args.epochs,
            batch_size        = args.batch,
            imgsz             = args.imgsz,
            device            = args.device,
            experiment_name   = args.name,
            project_dir       = args.project,
        )
    cfg.summary()
    cfg.save("temp/last_train_config.json")

    # ── 2. 데이터셋 검증 ────────────────────────────────────────────── #
    if not args.skip_val:
        import yaml
        with open(cfg.dataset_yaml, "r") as f:
            ds_meta = yaml.safe_load(f)
        dataset_root = ds_meta.get("path", ".")
        num_classes  = ds_meta.get("nc", 1)

        print("\n[main] 데이터셋 검증 시작...")
        validator = DatasetValidator(dataset_root=dataset_root, num_classes=num_classes)
        report    = validator.validate(dataset_yaml=cfg.dataset_yaml)
        validator.print_report()
        validator.save_report("temp/validation_report.json")

        if report["errors"]:
            print("\n 데이터셋 오류가 있습니다. 학습을 중단합니다.")
            print("   오류를 수정하거나 --skip-val 로 검증을 건너뛰세요.")
            sys.exit(1)
    else:
        print("[main] 데이터셋 검증 생략 (--skip-val)")

    # ── 3. 트레이너 초기화 ──────────────────────────────────────────── #
    trainer = ModelTrainer(cfg)

    resume_ckpt = None
    if args.resume:
        resume_ckpt = trainer.find_last_checkpoint()
        if resume_ckpt:
            print(f"[main] 재개 체크포인트 발견: {resume_ckpt}")
        else:
            print("[main] 재개할 체크포인트 없음 — 신규 학습으로 시작")

    trainer.setup(resume_checkpoint=resume_ckpt)

    # ── 4. 학습 실행 ────────────────────────────────────────────────── #
    summary = trainer.train()

    # ── 5. 결과 출력 ────────────────────────────────────────────────── #
    print("\n" + "=" * 50)
    print("  학습 완료")
    print("=" * 50)
    metrics = summary.get("metrics", {})
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print(f"\n  최적 가중치: {summary['best_weights']}")
    print("=" * 50 + "\n")


# ====================================================================== #
if __name__ == "__main__":
    main()