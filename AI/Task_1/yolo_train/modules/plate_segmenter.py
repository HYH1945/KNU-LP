"""
PlateSegmenter
- YOLOv8/v11-seg 모델로 원본 프레임에서 번호판 영역을 탐지하고
  픽셀 단위 Mask 배열을 추출하는 모듈
- 한 번에 한 프레임만 처리하여 OOM 방지
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ultralytics import YOLO


@dataclass
class PlateDetection:
    """
    단일 번호판 탐지 결과 컨테이너.

    Fields:
        mask_full  (ndarray) : 원본 해상도 기준 이진 마스크 (H, W), uint8 0/255
        mask_poly  (ndarray) : 마스크의 폴리곤 좌표 (N, 2), float32 픽셀 단위
        bbox_xyxy  (ndarray) : 바운딩박스 [x1, y1, x2, y2], float32
        conf       (float)   : 탐지 신뢰도 0~1
        class_id   (int)     : 클래스 인덱스
    """
    mask_full : np.ndarray
    mask_poly : np.ndarray
    bbox_xyxy : np.ndarray
    conf      : float
    class_id  : int


class PlateSegmenter:
    """
    YOLOv8/v11-seg 기반 번호판 Instance Segmentation 모듈.

    사용 예시:
        segmenter = PlateSegmenter(weights="best.pt", device="cuda")
        detections = segmenter.detect(frame)
        for det in detections:
            cv2.imshow("mask", det.mask_full)
    """

    def __init__(
        self,
        weights : str,
        device  : str  = "cuda",
        conf_thr: float = 0.25,
        iou_thr : float = 0.45,
        imgsz   : int   = 640,
    ):
        """
        Input:
            weights  (str)   : YOLO seg .pt 가중치 경로
            device   (str)   : "cuda", "cpu", "0"
            conf_thr (float) : 탐지 신뢰도 임계값
            iou_thr  (float) : NMS IoU 임계값
            imgsz    (int)   : 추론 입력 해상도
        """
        if not Path(weights).exists():
            raise FileNotFoundError(f"가중치 파일 없음: {weights}")

        self.conf_thr = conf_thr
        self.iou_thr  = iou_thr
        self.imgsz    = imgsz

        # CUDA 없으면 자동으로 CPU 전환
        if device in ("cuda", "0") and not torch.cuda.is_available():
            print("[PlateSegmenter] CUDA 없음 → CPU로 전환")
            device = "cpu"

        self.device = device
        self.model  = YOLO(weights)
        self.model.to(device)

        print(f"[PlateSegmenter] 로드 완료: {weights} | device={device}")

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        """
        단일 프레임에서 번호판을 탐지하고 마스크를 추출한다.

        Input:
            frame (ndarray) : BGR 원본 이미지 (H, W, 3), uint8
        Output:
            detections (list[PlateDetection]) : 탐지된 번호판 목록.
                                                탐지 없으면 빈 리스트.
        """
        if frame is None or frame.size == 0:
            return []

        H, W = frame.shape[:2]

        # 추론 (한 프레임, 그래디언트 계산 없음)
        with torch.no_grad():
            results = self.model.predict(
                source   = frame,
                conf     = self.conf_thr,
                iou      = self.iou_thr,
                imgsz    = self.imgsz,
                device   = self.device,
                verbose  = False,
                retina_masks = True,   # 원본 해상도 마스크 반환
            )

        result = results[0]

        # 마스크 없으면 빈 리스트
        if result.masks is None or len(result.masks) == 0:
            return []

        detections = []
        for i in range(len(result.masks)):
            det = self._extract_detection(result, i, H, W)
            if det is not None:
                detections.append(det)

        # GPU 메모리 즉시 해제
        del results, result
        if self.device != "cpu":
            torch.cuda.empty_cache()

        return detections

    # ------------------------------------------------------------------ #
    def _extract_detection(
        self,
        result,
        idx   : int,
        orig_H: int,
        orig_W: int,
    ) -> Optional[PlateDetection]:
        """
        단일 탐지 인덱스에서 PlateDetection 객체를 생성한다.

        Input:
            result        : ultralytics Results 객체
            idx    (int)  : 탐지 인덱스
            orig_H (int)  : 원본 이미지 높이
            orig_W (int)  : 원본 이미지 너비
        Output:
            PlateDetection 또는 None (마스크 추출 실패 시)
        """
        try:
            # ── 마스크 추출 ────────────────────────────────────────────
            # retina_masks=True → result.masks.data는 이미 원본 해상도
            mask_tensor = result.masks.data[idx]           # Tensor (H, W)
            mask_np     = mask_tensor.cpu().numpy()        # float32 0~1

            # 원본 해상도와 다를 경우 리사이즈
            if mask_np.shape != (orig_H, orig_W):
                mask_np = cv2.resize(
                    mask_np, (orig_W, orig_H),
                    interpolation=cv2.INTER_LINEAR
                )

            # 이진화
            mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255

            # ── 폴리곤 좌표 추출 ───────────────────────────────────────
            mask_poly = self._mask_to_polygon(mask_uint8, orig_H, orig_W)
            if mask_poly is None:
                return None

            # ── BBox 추출 ──────────────────────────────────────────────
            bbox = result.boxes.xyxy[idx].cpu().numpy().astype(np.float32)

            # ── 신뢰도 / 클래스 ───────────────────────────────────────
            conf     = float(result.boxes.conf[idx].cpu())
            class_id = int(result.boxes.cls[idx].cpu())

            return PlateDetection(
                mask_full = mask_uint8,
                mask_poly = mask_poly,
                bbox_xyxy = bbox,
                conf      = conf,
                class_id  = class_id,
            )

        except Exception as e:
            print(f"[PlateSegmenter] 탐지 {idx} 추출 실패: {e}")
            return None

    # ------------------------------------------------------------------ #
    def _mask_to_polygon(
        self,
        mask_uint8: np.ndarray,
        orig_H    : int,
        orig_W    : int,
    ) -> Optional[np.ndarray]:
        """
        이진 마스크에서 최대 면적 외곽선 폴리곤 좌표를 반환한다.

        Input:
            mask_uint8 (ndarray) : 이진 마스크 (H, W), uint8 0/255
            orig_H     (int)     : 원본 높이 (범위 클리핑용)
            orig_W     (int)     : 원본 너비 (범위 클리핑용)
        Output:
            polygon (ndarray) : 외곽선 좌표 (N, 2), float32 픽셀 단위
                                실패 시 None
        """
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # 가장 큰 외곽선 선택
        contour = max(contours, key=cv2.contourArea)

        # 너무 작은 영역 필터 (전체 면적의 0.01% 미만)
        min_area = orig_H * orig_W * 0.0001
        if cv2.contourArea(contour) < min_area:
            return None

        polygon = contour.reshape(-1, 2).astype(np.float32)

        # 좌표 범위 클리핑
        polygon[:, 0] = np.clip(polygon[:, 0], 0, orig_W - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, orig_H - 1)

        return polygon

    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        """
        더미 이미지로 모델 워밍업 (첫 추론 지연 방지).

        Input : (없음)
        Output: (없음)
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[PlateSegmenter] 워밍업 완료")


# ====================================================================== #
if __name__ == "__main__":
    """
    단독 실행 테스트:
    - 실제 가중치 경로와 이미지 경로를 수정 후 실행
    - 탐지 결과 시각화 및 마스크 저장
    """
    import sys
    import os

    WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "runs/train/plate_seg_nano/weights/best.pt"
    IMG_DIR = sys.argv[2] if len(sys.argv) > 2 else "test_images"

    if not Path(WEIGHTS).exists():
        print(f"가중치 없음: {WEIGHTS}")
        print("사용법: python plate_segmenter.py [weights.pt] [image_dir]")
        sys.exit(1)

    segmenter = PlateSegmenter(weights=WEIGHTS, device="cuda", conf_thr=0.25)
    segmenter.warmup()

    img_paths = sorted([
        p for p in Path(IMG_DIR).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not img_paths:
        print(f"이미지 없음: {IMG_DIR}")
        sys.exit(1)

    os.makedirs("test_output", exist_ok=True)

    for img_path in img_paths[:5]:   # 최대 5장 테스트
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        detections = segmenter.detect(frame)
        print(f"\n{img_path.name}: {len(detections)}개 탐지")

        vis = frame.copy()
        for i, det in enumerate(detections):
            print(f"  [{i}] conf={det.conf:.3f} | bbox={det.bbox_xyxy.astype(int)}")

            # 마스크 오버레이 (초록색)
            overlay        = vis.copy()
            overlay[det.mask_full > 0] = [0, 255, 0]
            vis            = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

            # BBox 그리기
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"{det.conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 폴리곤 외곽선 그리기 (파란색)
            pts = det.mask_poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        out_path = f"test_output/{img_path.stem}_result.jpg"
        cv2.imwrite(out_path, vis)
        print(f"  저장: {out_path}")

    print("\n PlateSegmenter 단독 테스트 완료")
    print("   결과 이미지: test_output/")
