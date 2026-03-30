"""
TrainConfig
- 학습 전반의 하이퍼파라미터 및 경로를 단일 객체로 관리하는 설정 모듈
- 모든 트레이너 모듈은 이 Config만 주입받아 동작한다
"""

from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class TrainConfig:
    """
    YOLO segmentation 학습 전체 설정.

    Input  (생성 시): 각 필드 값
    Output (사용 시): trainer.py에 주입되는 불변 설정 객체
    """

    # ── 경로 ──────────────────────────────────────────────────────────── #
    dataset_yaml   : str  = "dataset.yaml"          # 데이터셋 YAML 경로
    project_dir    : str  = "runs/segment"             # 학습 결과 저장 루트
    experiment_name: str  = "plate_seg"              # 실험 이름 (하위 폴더)
    pretrained_weights: str = "yolov8n-seg.pt"       # 사전학습 가중치
                                                     # yolov8{n/s/m/l/x}-seg.pt
                                                     # yolo11{n/s/m/l/x}-seg.pt

    # ── 학습 기본 파라미터 ────────────────────────────────────────────── #
    epochs         : int   = 100
    batch_size     : int   = 16                      # OOM 시 8로 낮춤
    imgsz          : int   = 640                     # 입력 해상도
    device         : str   = "0"                     # "0"=GPU0, "cpu", "0,1"=멀티GPU
    workers        : int   = 4                       # DataLoader 워커 수

    # ── 옵티마이저 ───────────────────────────────────────────────────── #
    optimizer      : str   = "AdamW"                 # SGD | Adam | AdamW
    lr0            : float = 1e-3                    # 초기 학습률
    lrf            : float = 1e-2                    # 최종 학습률 비율 (lr0 * lrf)
    momentum       : float = 0.937
    weight_decay   : float = 5e-4

    # ── 학습률 스케줄러 ──────────────────────────────────────────────── #
    warmup_epochs  : float = 3.0
    warmup_momentum: float = 0.8
    cos_lr         : bool  = True                    # Cosine LR decay

    # ── 손실 가중치 ──────────────────────────────────────────────────── #
    box            : float = 7.5                     # Box regression loss
    cls            : float = 0.5                     # Classification loss
    seg            : float = 1.0                     # Segmentation mask loss

    # ── Augmentation ─────────────────────────────────────────────────── #
    hsv_h          : float = 0.015                   # Hue jitter
    hsv_s          : float = 0.7                     # Saturation jitter
    hsv_v          : float = 0.4                     # Value jitter
    degrees        : float = 5.0                     # 회전 범위 (번호판: 소각도)
    translate      : float = 0.1
    scale          : float = 0.5
    shear          : float = 2.0                     # 전단 변환
    perspective    : float = 0.0005                  # 원근 왜곡 (핵심 증강)
    flipud         : float = 0.0                     # 번호판: 상하 반전 OFF
    fliplr         : float = 0.0                     # 번호판: 좌우 반전 OFF
    mosaic         : float = 1.0
    mixup          : float = 0.1
    copy_paste     : float = 0.1                     # Seg 전용 Copy-Paste

    # ── 검증 및 저장 ─────────────────────────────────────────────────── #
    val_period     : int   = 1                       # 매 N 에폭마다 검증
    save_period    : int   = 10                      # 매 N 에폭마다 체크포인트 저장
    patience       : int   = 30                      # Early stopping patience
    exist_ok       : bool  = True                    # 실험 폴더 덮어쓰기 허용

    # ── 기타 ─────────────────────────────────────────────────────────── #
    seed           : int   = 42
    verbose        : bool  = True
    plots          : bool  = True                    # 학습 곡선, confusion matrix 저장

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """
        Input : (없음)
        Output: config 전체를 dict로 변환 (ultralytics trainer 주입용)
        """
        return asdict(self)

    def save(self, path: str = "train_config.json") -> None:
        """
        Input:
            path (str) : 저장할 JSON 파일 경로
        Output: (없음, 파일 저장)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[TrainConfig] 설정 저장 완료 → {path}")

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        """
        Input:
            path (str) : 불러올 JSON 파일 경로
        Output:
            TrainConfig 인스턴스
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> None:
        """
        Input : (없음)
        Output: 주요 설정값 콘솔 출력
        """
        print("\n" + "=" * 50)
        print("  TrainConfig Summary")
        print("=" * 50)
        highlight = [
            "pretrained_weights", "dataset_yaml", "epochs",
            "batch_size", "imgsz", "device", "optimizer",
            "lr0", "patience", "experiment_name"
        ]
        for k in highlight:
            print(f"  {k:<22}: {getattr(self, k)}")
        print("=" * 50 + "\n")


# ====================================================================== #
if __name__ == "__main__":
    cfg = TrainConfig(
        pretrained_weights="yolov8s-seg.pt",
        epochs=50,
        batch_size=8,
        device="0",
    )
    cfg.summary()
    cfg.save("temp/train_config.json")

    # 저장 후 재로드 테스트
    loaded = TrainConfig.load("temp/train_config.json")
    assert loaded.epochs == 50
    print(" save/load 테스트 통과")
