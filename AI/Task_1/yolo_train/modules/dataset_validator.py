"""
DatasetValidator
- YOLO segmentation 학습 전 데이터셋 구조 및 라벨 무결성 검증 모듈
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class DatasetValidator:
    """
    YOLO segmentation 포맷 데이터셋의 구조, 라벨, 이미지 무결성을 검증한다.

    Expected directory structure:
        dataset_root/
        ├── images/
        │   ├── train/   *.jpg, *.png, ...
        │   └── val/
        └── labels/
            ├── train/   *.txt  (YOLO seg polygon format)
            └── val/
    """

    SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, dataset_root: str, num_classes: int = 1):
        """
        Input:
            dataset_root (str) : 데이터셋 루트 경로
            num_classes  (int) : 클래스 수 (번호판 단일 클래스 기본값 1)
        """
        self.root = Path(dataset_root)
        self.num_classes = num_classes
        self.report: dict = {}

    # ------------------------------------------------------------------ #
    def validate(self, dataset_yaml: str | None = None) -> dict:
        """
        전체 검증 실행.

        Input:
            dataset_yaml (str | None) : dataset.yaml 경로.
                                        지정 시 yaml의 train/val 경로를 직접 사용.
                                        None이면 train/images + valid/images 구조 자동 탐지.
        Output: report (dict) — 검증 결과 요약
            {
                "structure_ok"  : bool,
                "splits"        : { "train": {...}, "val": {...} },
                "total_images"  : int,
                "total_labels"  : int,
                "errors"        : list[str],
                "warnings"      : list[str],
            }
        """
        errors, warnings = [], []

        # 1. yaml 또는 자동 탐지로 실제 경로 결정
        split_dirs = self._resolve_split_dirs(dataset_yaml)

        # 2. 디렉토리 존재 여부 확인
        structure_ok = True
        for split, dirs in split_dirs.items():
            for d in [dirs["img"], dirs["label"]]:
                if not d.exists():
                    errors.append(f"[MISSING DIR] {d}")
                    structure_ok = False

        # 3. 분할별 상세 검증
        split_reports = {}
        for split, dirs in split_dirs.items():
            if dirs["img"].exists() and dirs["label"].exists():
                split_report, s_errors, s_warnings = self._validate_split(
                    dirs["img"], dirs["label"], split
                )
                split_reports[split] = split_report
                errors.extend(s_errors)
                warnings.extend(s_warnings)

        total_images = sum(r.get("image_count", 0) for r in split_reports.values())
        total_labels = sum(r.get("label_count", 0) for r in split_reports.values())

        self.report = {
            "structure_ok" : structure_ok and len(errors) == 0,
            "splits"       : split_reports,
            "total_images" : total_images,
            "total_labels" : total_labels,
            "errors"       : errors,
            "warnings"     : warnings,
        }
        return self.report

    # ------------------------------------------------------------------ #
    def _resolve_split_dirs(self, dataset_yaml: str | None) -> dict:
        """
        yaml 또는 자동 탐지를 통해 실제 이미지/라벨 디렉토리 경로를 결정.

        Input:
            dataset_yaml (str | None) : dataset.yaml 경로
        Output:
            {
              "train": {"img": Path, "label": Path},
              "val"  : {"img": Path, "label": Path},
            }
        """
        import yaml as _yaml

        if dataset_yaml and Path(dataset_yaml).exists():
            with open(dataset_yaml, "r") as f:
                meta = _yaml.safe_load(f)
            base = Path(meta.get("path", str(self.root)))
            train_img = base / meta.get("train", "train/images")
            val_img   = base / meta.get("val",   "valid/images")
        else:
            # yaml 없으면 일반적인 두 가지 구조 자동 탐지
            # 구조 A: train/images  (실제 사용 중인 구조)
            # 구조 B: images/train  (기존 기본값)
            if (self.root / "train" / "images").exists():
                train_img = self.root / "train" / "images"
                val_img   = (self.root / "valid" / "images"
                             if (self.root / "valid" / "images").exists()
                             else self.root / "val" / "images")
            else:
                train_img = self.root / "images" / "train"
                val_img   = self.root / "images" / "val"

        train_lbl = Path(str(train_img).replace("images", "labels"))
        val_lbl   = Path(str(val_img).replace("images", "labels"))

        return {
            "train": {"img": train_img, "label": train_lbl},
            "val"  : {"img": val_img,   "label": val_lbl},
        }

    # ------------------------------------------------------------------ #
    def _validate_split(
        self, img_dir: Path, label_dir: Path, split: str
    ) -> tuple[dict, list, list]:
        """
        Input:
            img_dir   (Path) : 이미지 디렉토리
            label_dir (Path) : 라벨 디렉토리
            split     (str)  : "train" 또는 "val"
        Output:
            (report: dict, errors: list[str], warnings: list[str])
        """
        errors, warnings = [], []

        images = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_IMG_EXTS
        ])
        labels = sorted(list(label_dir.glob("*.txt")))

        img_stems   = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        # 쌍 불일치 탐지
        only_images = img_stems - label_stems
        only_labels = label_stems - img_stems
        for s in only_images:
            warnings.append(f"[{split}] 라벨 없는 이미지: {s}")
        for s in only_labels:
            errors.append(f"[{split}] 이미지 없는 라벨: {s}")

        # 라벨 포맷 검증
        broken_labels, empty_labels, polygon_stats = [], [], []
        for lp in labels:
            ok, reason, poly_count = self._validate_label_file(lp)
            if not ok:
                broken_labels.append(f"{lp.name}: {reason}")
            if poly_count == 0:
                empty_labels.append(lp.name)
            else:
                polygon_stats.append(poly_count)

        for msg in broken_labels:
            errors.append(f"[{split}] 손상 라벨 — {msg}")
        for name in empty_labels:
            warnings.append(f"[{split}] 빈 라벨 파일: {name}")

        # 이미지 읽기 가능 여부 샘플 검사 (최대 20장)
        unreadable = []
        for ip in images[:20]:
            if cv2.imread(str(ip)) is None:
                unreadable.append(ip.name)
        for name in unreadable:
            errors.append(f"[{split}] 읽기 불가 이미지: {name}")

        report = {
            "image_count"    : len(images),
            "label_count"    : len(labels),
            "matched_pairs"  : len(img_stems & label_stems),
            "broken_labels"  : len(broken_labels),
            "empty_labels"   : len(empty_labels),
            "avg_polygons"   : float(np.mean(polygon_stats)) if polygon_stats else 0.0,
        }
        return report, errors, warnings

    # ------------------------------------------------------------------ #
    def _validate_label_file(self, label_path: Path) -> tuple[bool, str, int]:
        """
        Input:
            label_path (Path) : .txt 라벨 파일 경로
        Output:
            (is_valid: bool, reason: str, polygon_count: int)
            YOLO seg 포맷: 각 줄 = "class_id x1 y1 x2 y2 ... xn yn" (정규화 좌표)
        """
        try:
            lines = label_path.read_text().strip().splitlines()
            polygon_count = 0
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                tokens = line.split()
                # 최소: class_id + 3 꼭짓점(6개 좌표) = 7 토큰
                if len(tokens) < 7:
                    return False, f"line {i+1}: 토큰 부족 ({len(tokens)}개)", polygon_count
                class_id = int(tokens[0])
                if class_id >= self.num_classes:
                    return False, f"line {i+1}: class_id={class_id} 범위 초과", polygon_count
                coords = list(map(float, tokens[1:]))
                if len(coords) % 2 != 0:
                    return False, f"line {i+1}: 좌표 홀수개", polygon_count
                if not all(0.0 <= c <= 1.0 for c in coords):
                    return False, f"line {i+1}: 정규화 범위(0~1) 이탈", polygon_count
                polygon_count += 1
            return True, "", polygon_count
        except Exception as e:
            return False, str(e), 0

    # ------------------------------------------------------------------ #
    def print_report(self) -> None:
        """
        Input : (없음, self.report 사용)
        Output: 콘솔 출력 (없음)
        """
        if not self.report:
            print("[DatasetValidator] validate() 먼저 실행하세요.")
            return

        print("\n" + "=" * 55)
        print("  DatasetValidator Report")
        print("=" * 55)
        print(f"  구조 유효성  : {'✅ OK' if self.report['structure_ok'] else '❌ FAIL'}")
        print(f"  총 이미지    : {self.report['total_images']}")
        print(f"  총 라벨      : {self.report['total_labels']}")

        for split, r in self.report["splits"].items():
            print(f"\n  [{split}]")
            for k, v in r.items():
                print(f"    {k:<18}: {v}")

        if self.report["errors"]:
            print(f"\n  ❌ ERRORS ({len(self.report['errors'])})")
            for e in self.report["errors"]:
                print(f"    • {e}")

        if self.report["warnings"]:
            print(f"\n  ⚠️  WARNINGS ({len(self.report['warnings'])})")
            for w in self.report["warnings"]:
                print(f"    • {w}")

        print("=" * 55 + "\n")

    # ------------------------------------------------------------------ #
    def save_report(self, save_path: str = "validation_report.json") -> None:
        """
        Input:
            save_path (str) : 저장할 JSON 파일 경로
        Output: (없음, 파일 저장)
        """
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"[DatasetValidator] 리포트 저장 완료 → {save_path}")


# ====================================================================== #
if __name__ == "__main__":
    import sys

    # 경로를 인자로 받거나 기본값 사용
    root = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    validator = DatasetValidator(dataset_root=root, num_classes=1)
    report = validator.validate()
    validator.print_report()
    validator.save_report("validation_report.json")

    # 에러 있으면 비정상 종료 (CI/CD 파이프라인 연동 용도)
    if not report["structure_ok"] or report["errors"]:
        sys.exit(1)