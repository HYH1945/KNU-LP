"""
ImageLoader
- 지정된 입력 폴더 내 이미지 파일을 파일명 오름차순으로 정렬하여 경로 리스트를 반환
- 한 번에 한 프레임씩 로드하는 제너레이터 방식으로 OOM 방지
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


class ImageLoader:
    """
    특정 폴더 내 이미지 파일을 파일명 기준 오름차순으로 순차 로드.
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, input_dir: str):
        """
        Input:
            input_dir (str) : 이미지가 있는 폴더 경로
        """
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"입력 폴더 없음: {input_dir}")

        self._paths = self._collect_paths()

    # ------------------------------------------------------------------ #
    def _collect_paths(self) -> list[Path]:
        """
        Input : (없음, self.input_dir 사용)
        Output: 정렬된 이미지 파일 경로 리스트 (list[Path])
        """
        paths = sorted([
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        ])
        if not paths:
            raise FileNotFoundError(f"이미지 없음: {self.input_dir}")
        return paths

    # ------------------------------------------------------------------ #
    @property
    def paths(self) -> list[Path]:
        """
        Input : (없음)
        Output: 정렬된 이미지 파일 경로 리스트 (list[Path])
        """
        return self._paths

    @property
    def total(self) -> int:
        """
        Input : (없음)
        Output: 총 이미지 수 (int)
        """
        return len(self._paths)

    # ------------------------------------------------------------------ #
    def load(self, index: int) -> tuple[np.ndarray, Path]:
        """
        단일 인덱스의 이미지를 로드한다.

        Input:
            index (int) : 이미지 인덱스 (0-based)
        Output:
            (frame: ndarray, path: Path)
            frame — BGR 이미지 (H, W, 3), uint8
        """
        path  = self._paths[index]
        frame = cv2.imread(str(path))
        if frame is None:
            raise IOError(f"이미지 읽기 실패: {path}")
        return frame, path

    # ------------------------------------------------------------------ #
    def iter_frames(self, start_index: int = 0):
        """
        start_index부터 순차적으로 (index, frame, path)를 yield하는 제너레이터.
        한 번에 한 프레임만 메모리에 로드.

        Input:
            start_index (int) : 시작 인덱스 (재개 시 사용)
        Output:
            generator of (index: int, frame: ndarray, path: Path)
        """
        for idx in range(start_index, self.total):
            frame, path = self.load(idx)
            yield idx, frame, path
            del frame   # 즉시 메모리 해제


# ====================================================================== #
if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "input/imageset_1"
    loader    = ImageLoader(input_dir)

    print(f"총 이미지 수: {loader.total}")
    print("파일 목록:")
    for i, p in enumerate(loader.paths):
        print(f"  [{i}] {p.name}")

    # 첫 프레임 로드 테스트
    frame, path = loader.load(0)
    print(f"\n첫 프레임: {path.name} | shape={frame.shape}")
    print("✅ ImageLoader 단독 테스트 통과")
