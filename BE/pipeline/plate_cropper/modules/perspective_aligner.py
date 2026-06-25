"""
PerspectiveAligner
- PlateDetection의 Mask로부터 4점 좌표를 추출하고
  투시 변환(Perspective Transform)을 적용하여 정면 번호판 이미지를 반환
"""

import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass

from modules.plate_segmenter import PlateDetection


@dataclass
class AlignResult:
    """
    투시 변환 결과 컨테이너.

    Fields:
        warped      (ndarray)       : 보정된 번호판 이미지 (H, W, 3), uint8
        corners_src (ndarray)       : 원본 이미지 기준 4점 좌표 (4, 2), float32
                                      순서: 좌상, 우상, 우하, 좌하
        success     (bool)          : 변환 성공 여부
        reason      (str)           : 실패 사유 (성공 시 빈 문자열)
    """
    warped      : Optional[np.ndarray]
    corners_src : Optional[np.ndarray]
    success     : bool
    reason      : str = ""


class PerspectiveAligner:
    """
    번호판 Mask → 4점 추출 → 투시 변환 → 정면 이미지 반환.

    국내 표준 번호판 비율 기준으로 출력 크기 고정.
    """

    def __init__(self, output_width: int = 335, output_height: int = 170):
        """
        Input:
            output_width  (int) : 출력 이미지 너비 (기본 335 → 일반 승용 신형 기준)
            output_height (int) : 출력 이미지 높이 (기본 170)
        """
        self.out_w = output_width
        self.out_h = output_height

        # 목적지 4점 (좌상, 우상, 우하, 좌하)
        self._dst_pts = np.array([
            [0,           0          ],
            [output_width, 0          ],
            [output_width, output_height],
            [0,           output_height],
        ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    def align(self, frame: np.ndarray, detection: PlateDetection) -> AlignResult:
        """
        원본 프레임과 탐지 결과를 받아 투시 보정 번호판 이미지를 반환.

        Input:
            frame     (ndarray)       : BGR 원본 이미지 (H, W, 3), uint8
            detection (PlateDetection): PlateSegmenter 출력 탐지 결과
        Output:
            AlignResult
        """
        # 1. Mask → 4점 추출
        corners, reason = self._extract_corners(detection.mask_full)
        if corners is None:
            return AlignResult(warped=None, corners_src=None, success=False, reason=reason)

        # 2. 투시 변환 행렬 계산
        M = cv2.getPerspectiveTransform(corners, self._dst_pts)

        # 3. 원본 이미지에 변환 적용
        warped = cv2.warpPerspective(frame, M, (self.out_w, self.out_h))

        return AlignResult(
            warped      = warped,
            corners_src = corners,
            success     = True,
        )

    # ------------------------------------------------------------------ #
    def _extract_corners(
        self, mask_uint8: np.ndarray
    ) -> tuple[Optional[np.ndarray], str]:
        """
        이진 마스크에서 번호판 4점 꼭짓점을 추출하고 정렬한다.

        Input:
            mask_uint8 (ndarray) : 이진 마스크 (H, W), uint8 0/255
        Output:
            (corners: ndarray (4, 2) float32, reason: str)
            corners — 순서: 좌상, 우상, 우하, 좌하
            실패 시 (None, reason_str)
        """
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, "컨투어 없음"

        contour = max(contours, key=cv2.contourArea)

        # ── 다각형 근사 → 4점 시도 ────────────────────────────────── #
        corners = self._approx_to_quad(contour)

        # 근사 실패 시 Convex Hull로 폴백
        if corners is None:
            hull    = cv2.convexHull(contour)
            corners = self._approx_to_quad(hull)

        # 그래도 실패 시 BBox 4점으로 폴백
        if corners is None:
            corners = self._bbox_corners(contour)

        if corners is None:
            return None, "4점 추출 실패"

        # 좌표 정렬 (좌상, 우상, 우하, 좌하)
        ordered = self._order_corners(corners)
        return ordered, ""

    # ------------------------------------------------------------------ #
    def _approx_to_quad(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        다각형 근사를 통해 4점 사각형을 추출한다.
        epsilon을 동적으로 조절하며 시도.

        Input:
            contour (ndarray) : 외곽선 좌표 (N, 1, 2)
        Output:
            corners (ndarray) : 4점 좌표 (4, 2), float32
                                실패 시 None
        """
        arc_len = cv2.arcLength(contour, True)

        for eps_ratio in [0.02, 0.03, 0.04, 0.05, 0.06]:
            epsilon = eps_ratio * arc_len
            approx  = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        return None

    # ------------------------------------------------------------------ #
    def _bbox_corners(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        컨투어의 최소 외접 사각형(BBox) 4점을 반환한다. (최후 폴백)

        Input:
            contour (ndarray) : 외곽선 좌표 (N, 1, 2)
        Output:
            corners (ndarray) : 4점 좌표 (4, 2), float32
        """
        x, y, w, h = cv2.boundingRect(contour)
        return np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        4점을 [좌상, 우상, 우하, 좌하] 순서로 정렬한다.

        Input:
            pts (ndarray) : 임의 순서의 4점 좌표 (4, 2), float32
        Output:
            ordered (ndarray) : 정렬된 4점 좌표 (4, 2), float32
        """
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        ordered = np.array([
            pts[np.argmin(s)],     # 좌상: x+y 최소
            pts[np.argmin(diff)],  # 우상: y-x 최소
            pts[np.argmax(s)],     # 우하: x+y 최대
            pts[np.argmax(diff)],  # 좌하: y-x 최대
        ], dtype=np.float32)

        return ordered


# ====================================================================== #
if __name__ == "__main__":
    """
    단독 실행 테스트:
    - 임의 사다리꼴 마스크를 생성하여 투시 변환 결과 검증
    """
    import sys

    # 더미 프레임 및 사다리꼴 마스크 생성
    H, W  = 480, 640
    frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    mask  = np.zeros((H, W), dtype=np.uint8)
    quad  = np.array([[120, 200], [400, 180], [420, 280], [100, 300]])
    cv2.fillPoly(mask, [quad], 255)

    # PlateDetection 더미 객체
    from plate_segmenter import PlateDetection
    det = PlateDetection(
        mask_full = mask,
        mask_poly = quad.astype(np.float32),
        bbox_xyxy = np.array([100, 180, 420, 300], dtype=np.float32),
        conf      = 0.95,
        class_id  = 0,
    )

    aligner = PerspectiveAligner(output_width=335, output_height=170)
    result  = aligner.align(frame, det)

    assert result.success, f"투시 변환 실패: {result.reason}"
    assert result.warped.shape == (170, 335, 3), f"출력 shape 오류: {result.warped.shape}"
    assert result.corners_src.shape == (4, 2), "4점 좌표 shape 오류"

    print(f"✅ PerspectiveAligner 단독 테스트 통과")
    print(f"   warped shape : {result.warped.shape}")
    print(f"   corners_src  :\n{result.corners_src}")

    cv2.imwrite("test_warped.png", result.warped)
    print("   저장: test_warped.png")
