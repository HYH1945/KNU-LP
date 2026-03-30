"""
FormatConverter
- COCO JSON / LabelMe JSON 형식의 라벨을 YOLO segmentation .txt 포맷으로 변환하는 모듈
- 변환 후 dataset/ 디렉토리 구조를 자동 생성
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict


class FormatConverter:
    """
    외부 라벨링 포맷 → YOLO seg (.txt) 변환기.

    지원 포맷:
        - COCO Instance Segmentation JSON
        - LabelMe JSON (폴더 내 복수 파일)
    """

    def __init__(self, output_root: str = "./dataset"):
        """
        Input:
            output_root (str) : 변환 결과를 저장할 루트 경로
                                구조: output_root/images/{train,val}/
                                      output_root/labels/{train,val}/
        """
        self.output_root = Path(output_root)

    # ================================================================== #
    #  COCO → YOLO seg
    # ================================================================== #
    def from_coco(
        self,
        coco_json_path: str,
        images_dir: str,
        split: str = "train",
        class_names: list[str] | None = None,
    ) -> dict:
        """
        COCO Instance Segmentation JSON을 YOLO seg 포맷으로 변환.

        Input:
            coco_json_path (str)              : COCO JSON 파일 경로
            images_dir     (str)              : 원본 이미지 폴더 경로
            split          (str)              : "train" 또는 "val"
            class_names    (list[str] | None) : 변환할 클래스명 리스트.
                                               None이면 전체 클래스 변환.
        Output:
            result (dict) — {"converted": int, "skipped": int, "class_map": dict}
        """
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # 카테고리 맵 구성
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        if class_names:
            valid_cat_ids = {cid for cid, name in cat_id_to_name.items() if name in class_names}
            class_map = {cid: i for i, cid in enumerate(sorted(valid_cat_ids))}
        else:
            class_map = {c["id"]: i for i, c in enumerate(coco["categories"])}
            valid_cat_ids = set(class_map.keys())

        # 이미지 ID → 정보 매핑
        img_id_to_info = {img["id"]: img for img in coco["images"]}

        # 어노테이션을 이미지 ID 기준으로 그룹핑
        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in valid_cat_ids and ann.get("segmentation"):
                ann_by_image[ann["image_id"]].append(ann)

        out_img_dir   = self._make_split_dirs(split)["img"]
        out_label_dir = self._make_split_dirs(split)["label"]

        converted, skipped = 0, 0

        for img_id, anns in ann_by_image.items():
            img_info = img_id_to_info.get(img_id)
            if not img_info:
                skipped += 1
                continue

            W, H = img_info["width"], img_info["height"]
            img_filename = img_info["file_name"]
            img_src = Path(images_dir) / img_filename

            # 이미지 복사
            if img_src.exists():
                shutil.copy2(img_src, out_img_dir / Path(img_filename).name)
            else:
                skipped += 1
                continue

            # 라벨 생성
            label_lines = []
            for ann in anns:
                class_idx = class_map[ann["category_id"]]
                for seg in ann["segmentation"]:
                    if len(seg) < 6:   # 최소 3점
                        continue
                    coords = np.array(seg).reshape(-1, 2)
                    # 정규화
                    coords[:, 0] /= W
                    coords[:, 1] /= H
                    coords = np.clip(coords, 0.0, 1.0)
                    flat = " ".join(f"{c:.6f}" for c in coords.flatten())
                    label_lines.append(f"{class_idx} {flat}")

            stem = Path(img_filename).stem
            label_path = out_label_dir / f"{stem}.txt"
            label_path.write_text("\n".join(label_lines))
            converted += 1

        result = {
            "converted" : converted,
            "skipped"   : skipped,
            "class_map" : {cat_id_to_name[cid]: idx for cid, idx in class_map.items()},
        }
        print(f"[COCO→YOLO] split={split} | 변환: {converted} | 스킵: {skipped}")
        return result

    # ================================================================== #
    #  LabelMe → YOLO seg
    # ================================================================== #
    def from_labelme(
        self,
        labelme_dir: str,
        split: str = "train",
        class_names: list[str] | None = None,
    ) -> dict:
        """
        LabelMe JSON 폴더를 YOLO seg 포맷으로 변환.

        Input:
            labelme_dir  (str)              : LabelMe JSON 파일들이 있는 폴더
            split        (str)              : "train" 또는 "val"
            class_names  (list[str] | None) : 필터링할 클래스명. None이면 전체.
        Output:
            result (dict) — {"converted": int, "skipped": int}
        """
        json_files = sorted(Path(labelme_dir).glob("*.json"))
        out_img_dir   = self._make_split_dirs(split)["img"]
        out_label_dir = self._make_split_dirs(split)["label"]

        converted, skipped = 0, 0
        discovered_classes: set[str] = set()

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            W = data.get("imageWidth")
            H = data.get("imageHeight")
            img_path_rel = data.get("imagePath", "")
            img_src = jf.parent / img_path_rel

            if not W or not H or not img_src.exists():
                skipped += 1
                continue

            shapes = [
                s for s in data.get("shapes", [])
                if s["shape_type"] == "polygon"
                and (class_names is None or s["label"] in class_names)
            ]

            if not shapes:
                skipped += 1
                continue

            # 클래스 인덱스 동적 할당
            for s in shapes:
                discovered_classes.add(s["label"])

            # 이미지 복사
            shutil.copy2(img_src, out_img_dir / img_src.name)

            # 라벨 생성 (클래스 인덱스는 정렬된 순서 기준)
            sorted_classes = sorted(discovered_classes)
            label_lines = []
            for s in shapes:
                class_idx = sorted_classes.index(s["label"])
                pts = np.array(s["points"], dtype=np.float32)
                pts[:, 0] /= W
                pts[:, 1] /= H
                pts = np.clip(pts, 0.0, 1.0)
                flat = " ".join(f"{c:.6f}" for c in pts.flatten())
                label_lines.append(f"{class_idx} {flat}")

            label_path = out_label_dir / f"{jf.stem}.txt"
            label_path.write_text("\n".join(label_lines))
            converted += 1

        result = {
            "converted"       : converted,
            "skipped"         : skipped,
            "discovered_classes": sorted(discovered_classes),
        }
        print(f"[LabelMe→YOLO] split={split} | 변환: {converted} | 스킵: {skipped}")
        print(f"  발견된 클래스: {result['discovered_classes']}")
        return result

    # ================================================================== #
    #  dataset.yaml 자동 생성
    # ================================================================== #
    def generate_yaml(self, class_names: list[str], yaml_path: str = "dataset.yaml") -> None:
        """
        학습에 필요한 dataset.yaml 파일을 자동 생성.

        Input:
            class_names (list[str]) : 클래스 이름 목록 (인덱스 순서)
            yaml_path   (str)       : 저장할 YAML 파일 경로
        Output: (없음, 파일 저장)
        """
        abs_root = str(self.output_root.resolve())
        lines = [
            f"path: {abs_root}",
            f"train: images/train",
            f"val:   images/val",
            f"",
            f"nc: {len(class_names)}",
            f"names: {class_names}",
        ]
        Path(yaml_path).write_text("\n".join(lines))
        print(f"[FormatConverter] dataset.yaml 생성 완료 → {yaml_path}")
        print(f"  클래스: {class_names}")

    # ------------------------------------------------------------------ #
    def _make_split_dirs(self, split: str) -> dict:
        """
        Input:
            split (str) : "train" 또는 "val"
        Output:
            {"img": Path, "label": Path} — 생성된 디렉토리 경로
        """
        img_dir   = self.output_root / "images" / split
        label_dir = self.output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        return {"img": img_dir, "label": label_dir}


# ====================================================================== #
if __name__ == "__main__":
    """
    단독 실행 테스트: 더미 LabelMe JSON으로 변환 로직 검증
    """
    import tempfile, sys

    # ── 더미 LabelMe JSON 생성 ──────────────────────────────────────── #
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 더미 이미지 (1x1 PNG)
        img_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (tmp / "test.png").write_bytes(img_data)

        dummy = {
            "imageWidth" : 640,
            "imageHeight": 480,
            "imagePath"  : "test.png",
            "shapes": [{
                "label"     : "license_plate",
                "shape_type": "polygon",
                "points"    : [[100,50],[300,50],[300,150],[100,150]],
            }]
        }
        (tmp / "test.json").write_text(json.dumps(dummy))

        converter = FormatConverter(output_root=str(tmp / "output"))
        result    = converter.from_labelme(str(tmp), split="train")
        converter.generate_yaml(
            class_names=result["discovered_classes"],
            yaml_path=str(tmp / "dataset.yaml"),
        )

        label_file = tmp / "output" / "labels" / "train" / "test.txt"
        assert label_file.exists(), "라벨 파일 생성 실패"
        content = label_file.read_text()
        assert content.startswith("0 "), f"포맷 오류: {content}"
        print(f"\n FormatConverter 단독 테스트 통과")
        print(f"   생성된 라벨: {content}")