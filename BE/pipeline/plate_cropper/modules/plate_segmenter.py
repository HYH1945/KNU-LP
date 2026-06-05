"""
PlateSegmenter
- YOLOv8/v11-seg 모델로 원본 프레임에서 번호판 영역을 탐지하고
  픽셀 단위 Mask 배열을 추출하는 모듈
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
        mask_full (ndarray) : 원본 해상도 이진 마스크 (H, W), uint8 0/255
        mask_poly (ndarray) : 마스크 외곽선 폴리곤 좌표 (N, 2), float32 픽셀
        bbox_xyxy (ndarray) : 바운딩박스 [x1, y1, x2, y2], float32
        conf      (float)   : 탐지 신뢰도 0~1
        class_id  (int)     : 클래스 인덱스
    """
    mask_full : np.ndarray
    mask_poly : np.ndarray
    bbox_xyxy : np.ndarray
    conf      : float
    class_id  : int


class PlateSegmenter:
    """
    YOLOv8/v11-seg 기반 번호판 Instance Segmentation 모듈.
    """

    def __init__(
        self,
        weights : str,
        device  : str   = "cuda",
        conf_thr: float = 0.25,
        iou_thr : float = 0.45,
        imgsz   : int   = 640,
    ):
        """
        Input:
            weights  (str)   : YOLO seg .pt 가중치 경로
            device   (str)   : "cuda" | "cpu"
            conf_thr (float) : 탐지 신뢰도 임계값
            iou_thr  (float) : NMS IoU 임계값
            imgsz    (int)   : 추론 입력 해상도
        """
        if not Path(weights).exists():
            raise FileNotFoundError(f"가중치 파일 없음: {weights}")

        if device in ("cuda", "0") and not torch.cuda.is_available():
            print("[PlateSegmenter] CUDA 없음 → CPU 전환")
            device = "cpu"

        self.device   = device
        self.conf_thr = conf_thr
        self.iou_thr  = iou_thr
        self.imgsz    = imgsz
        self.model    = YOLO(weights)
        self.model.to(device)

        print(f"[PlateSegmenter] 로드 완료 | device={device} | conf={conf_thr}")

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

        with torch.no_grad():
            results = self.model.predict(
                source       = frame,
                conf         = self.conf_thr,
                iou          = self.iou_thr,
                imgsz        = self.imgsz,
                device       = self.device,
                verbose      = False,
                retina_masks = True,
            )

        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            del results, result
            return []

        detections = []
        for i in range(len(result.masks)):
            det = self._extract_detection(result, i, H, W)
            if det is not None:
                detections.append(det)

        del results, result
        if self.device != "cpu":
            torch.cuda.empty_cache()

        return detections

    # ------------------------------------------------------------------ #
    def _extract_detection(
        self, result, idx: int, orig_H: int, orig_W: int
    ) -> Optional[PlateDetection]:
        """
        단일 탐지 인덱스에서 PlateDetection 객체를 생성한다.

        Input:
            result         : ultralytics Results 객체
            idx    (int)   : 탐지 인덱스
            orig_H (int)   : 원본 이미지 높이
            orig_W (int)   : 원본 이미지 너비
        Output:
            PlateDetection 또는 None (마스크 추출 실패 시)
        """
        try:
            mask_tensor = result.masks.data[idx]
            mask_np     = mask_tensor.cpu().numpy()

            if mask_np.shape != (orig_H, orig_W):
                mask_np = cv2.resize(
                    mask_np, (orig_W, orig_H),
                    interpolation=cv2.INTER_LINEAR
                )

            mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
            mask_poly  = self._mask_to_polygon(mask_uint8, orig_H, orig_W)
            if mask_poly is None:
                return None

            bbox     = result.boxes.xyxy[idx].cpu().numpy().astype(np.float32)
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
        self, mask_uint8: np.ndarray, orig_H: int, orig_W: int
    ) -> Optional[np.ndarray]:
        """
        이진 마스크에서 최대 면적 외곽선 폴리곤 좌표를 반환한다.

        Input:
            mask_uint8 (ndarray) : 이진 마스크 (H, W), uint8 0/255
            orig_H     (int)     : 원본 높이
            orig_W     (int)     : 원본 너비
        Output:
            polygon (ndarray) : 외곽선 좌표 (N, 2), float32 픽셀 단위
                                실패 시 None
        """
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour  = max(contours, key=cv2.contourArea)
        min_area = orig_H * orig_W * 0.0001
        if cv2.contourArea(contour) < min_area:
            return None

        polygon       = contour.reshape(-1, 2).astype(np.float32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, orig_W - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, orig_H - 1)
        return polygon

    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        """
        Input : (없음)
        Output: (없음) — 더미 이미지로 첫 추론 지연 방지
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[PlateSegmenter] 워밍업 완료")


# ====================================================================== #
if __name__ == "__main__":
    import sys

    WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "weights/best.pt"
    IMG_DIR = sys.argv[2] if len(sys.argv) > 2 else "input/imageset_1"

    segmenter = PlateSegmenter(weights=WEIGHTS, device="cuda", conf_thr=0.25)
    segmenter.warmup()

    from image_loader import ImageLoader
    loader = ImageLoader(IMG_DIR)

    for idx, frame, path in loader.iter_frames():
        dets = segmenter.detect(frame)
        print(f"[{idx}] {path.name} → {len(dets)}개 탐지")
        for i, d in enumerate(dets):
            print(f"      [{i}] conf={d.conf:.3f} bbox={d.bbox_xyxy.astype(int)}")

    print("✅ PlateSegmenter 단독 테스트 완료")
