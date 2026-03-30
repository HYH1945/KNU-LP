"""
ModelTrainer
- TrainConfig를 주입받아 YOLOv8/v11-seg 학습 전 과정을 실행하는 모듈
- 재개(resume), 콜백, 학습 결과 요약을 포함
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO
# from ultralytics.utils.callbacks.base import DEFAULT_CALLBACKS

from modules.train_config import TrainConfig


class ModelTrainer:
    """
    YOLO segmentation 모델 학습 실행 클래스.

    주요 책임:
        - YOLO 모델 로드 (사전학습 / 재개)
        - TrainConfig 기반 학습 파라미터 주입
        - 에폭별 콜백 (손실 로깅, 체크포인트)
        - 학습 완료 후 결과 요약 저장
    """

    def __init__(self, config: TrainConfig):
        """
        Input:
            config (TrainConfig) : 전체 학습 설정 객체
        """
        self.cfg   = config
        self.model : YOLO | None = None
        self.results = None
        self._resume_path: str | None = None

        os.makedirs("temp", exist_ok=True)

    # ------------------------------------------------------------------ #
    def setup(self, resume_checkpoint: str | None = None) -> None:
        """
        모델 초기화 또는 체크포인트로부터 재개 설정.

        Input:
            resume_checkpoint (str | None) : 재개할 .pt 파일 경로.
                                             None이면 사전학습 가중치로 신규 학습.
        Output: (없음, self.model 초기화)
        """
        if resume_checkpoint and Path(resume_checkpoint).exists():
            print(f"[ModelTrainer] 체크포인트 재개: {resume_checkpoint}")
            self._resume_path = resume_checkpoint
            self.model = YOLO(resume_checkpoint)
        else:
            print(f"[ModelTrainer] 신규 학습: {self.cfg.pretrained_weights}")
            self.model = YOLO(self.cfg.pretrained_weights)

        self._log_device_info()

    # ------------------------------------------------------------------ #
    def train(self) -> dict:
        """
        학습 실행.

        Input : (없음, self.cfg 및 self.model 사용)
        Output: results_summary (dict) — 최종 mAP, 경로 등 요약
        """
        if self.model is None:
            raise RuntimeError("setup()을 먼저 호출하세요.")

        # ultralytics YOLO.train()에 전달할 파라미터 조립
        train_args = self._build_train_args()

        print("\n[ModelTrainer] 학습 시작")
        print(f"  실험명   : {self.cfg.experiment_name}")
        print(f"  에폭     : {self.cfg.epochs}")
        print(f"  배치     : {self.cfg.batch_size}")
        print(f"  디바이스 : {self.cfg.device}\n")

        # ── 핵심 학습 호출 ──────────────────────────────────────────── #
        self.results = self.model.train(**train_args)

        # 학습 결과 요약 저장
        summary = self._build_summary()
        self._save_summary(summary)

        return summary

    # ------------------------------------------------------------------ #
    def _build_train_args(self) -> dict:
        """
        Input : (없음, self.cfg 사용)
        Output: ultralytics YOLO.train()에 전달할 kwargs dict
        """
        cfg = self.cfg
        args = {
            # 경로
            "data"    : cfg.dataset_yaml,
            "project" : cfg.project_dir,
            "name"    : cfg.experiment_name,

            # 기본 파라미터
            "epochs"   : cfg.epochs,
            "batch"    : cfg.batch_size,
            "imgsz"    : cfg.imgsz,
            "device"   : cfg.device,
            "workers"  : cfg.workers,
            "seed"     : cfg.seed,
            "verbose"  : cfg.verbose,
            "plots"    : cfg.plots,
            "exist_ok" : cfg.exist_ok,

            # 옵티마이저
            "optimizer"     : cfg.optimizer,
            "lr0"           : cfg.lr0,
            "lrf"           : cfg.lrf,
            "momentum"      : cfg.momentum,
            "weight_decay"  : cfg.weight_decay,
            "cos_lr"        : cfg.cos_lr,

            # 스케줄러
            "warmup_epochs"  : cfg.warmup_epochs,
            "warmup_momentum": cfg.warmup_momentum,

            # 손실 가중치
            "box" : cfg.box,
            "cls" : cfg.cls,
            "dfl" : 1.5,    # 고정값 (seg에서 dfl 사용)

            # Augmentation
            "hsv_h"      : cfg.hsv_h,
            "hsv_s"      : cfg.hsv_s,
            "hsv_v"      : cfg.hsv_v,
            "degrees"    : cfg.degrees,
            "translate"  : cfg.translate,
            "scale"      : cfg.scale,
            "shear"      : cfg.shear,
            "perspective": cfg.perspective,
            "flipud"     : cfg.flipud,
            "fliplr"     : cfg.fliplr,
            "mosaic"     : cfg.mosaic,
            "mixup"      : cfg.mixup,
            "copy_paste" : cfg.copy_paste,

            # 검증 / 저장
            "val"         : True,
            "save"        : True,
            "save_period" : cfg.save_period,
            "patience"    : cfg.patience,
        }

        # 재개 학습인 경우
        if self._resume_path:
            args["resume"] = True

        return args

    # ------------------------------------------------------------------ #
    def _build_summary(self) -> dict:
        """
        Input : (없음, self.results 사용)
        Output: 학습 완료 요약 dict
        """
        save_dir = Path(self.cfg.project_dir) / self.cfg.experiment_name

        summary = {
            "experiment" : self.cfg.experiment_name,
            "timestamp"  : datetime.now().isoformat(),
            "save_dir"   : str(save_dir),
            "best_weights": str(save_dir / "weights" / "best.pt"),
            "last_weights": str(save_dir / "weights" / "last.pt"),
        }

        # ultralytics results 객체에서 최종 메트릭 추출
        if self.results is not None:
            try:
                metrics = self.results.results_dict
                summary["metrics"] = {
                    "mAP50_seg"   : round(metrics.get("metrics/mAP50(M)",   0.0), 4),
                    "mAP50-95_seg": round(metrics.get("metrics/mAP50-95(M)",0.0), 4),
                    "mAP50_box"   : round(metrics.get("metrics/mAP50(B)",   0.0), 4),
                    "precision"   : round(metrics.get("metrics/precision(B)",0.0), 4),
                    "recall"      : round(metrics.get("metrics/recall(B)",   0.0), 4),
                }
            except Exception:
                summary["metrics"] = {}

        return summary

    # ------------------------------------------------------------------ #
    def _save_summary(self, summary: dict) -> None:
        """
        Input:
            summary (dict) : 학습 결과 요약
        Output: (없음, runs/train/{name}/train_summary.json 저장)
        """
        save_path = Path(summary["save_dir"]) / "train_summary.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[ModelTrainer] 학습 요약 저장 → {save_path}")

    # ------------------------------------------------------------------ #
    def _log_device_info(self) -> None:
        """
        Input : (없음)
        Output: 디바이스 정보 콘솔 출력
        """
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"[Device] GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
        else:
            print("[Device] CUDA 없음 — CPU 학습")

    # ------------------------------------------------------------------ #
    def find_last_checkpoint(self) -> str | None:
        """
        학습 중단 시 재개를 위해 마지막 저장된 체크포인트 경로를 반환.

        Input : (없음, self.cfg 사용)
        Output: checkpoint 경로 (str) 또는 None
        """
        last_pt = Path(self.cfg.project_dir) / self.cfg.experiment_name / "weights" / "last.pt"
        return str(last_pt) if last_pt.exists() else None


# ====================================================================== #
if __name__ == "__main__":
    """
    단독 실행 테스트:
    - 실제 학습 없이 setup / build_args 로직만 검증
    """
    cfg = TrainConfig(
        dataset_yaml      = "dataset.yaml",
        pretrained_weights= "yolov8n-seg.pt",
        epochs            = 2,
        batch_size        = 2,
        device            = "cpu",
        experiment_name   = "test_run",
    )
    cfg.summary()

    trainer = ModelTrainer(cfg)
    trainer.setup()

    args = trainer._build_train_args()
    print("\n[단독 테스트] _build_train_args 결과:")
    for k, v in args.items():
        print(f"  {k:<18}: {v}")
    print("\n ModelTrainer 초기화 및 args 빌드 테스트 통과")
