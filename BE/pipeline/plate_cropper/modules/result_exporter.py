"""
ResultExporter
- 보정된 번호판 이미지를 PNG로 저장하고 result_log.json을 기록하는 모듈
- 처리 완료 후 temp/ 폴더 정리
"""

import cv2
import json
import os
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime


class ResultExporter:
    """
    투시 보정 결과 이미지 저장 및 로그 기록 모듈.
    """

    def __init__(self, output_dir: str, prefix: str = "aligned_frame_"):
        """
        Input:
            output_dir (str) : 결과 저장 폴더 경로
            prefix     (str) : 출력 파일명 접두사
        """
        self.output_dir = Path(output_dir)
        self.prefix     = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._log: list[dict] = []

    # ------------------------------------------------------------------ #
    def save(
        self,
        warped      : np.ndarray,
        frame_index : int,
        src_filename: str,
        corners_src : np.ndarray,
        conf        : float,
        det_index   : int = 0,
    ) -> bool:
        """
        보정 이미지를 PNG로 저장하고 로그에 기록한다.

        Input:
            warped       (ndarray) : 보정된 번호판 이미지 (H, W, 3), uint8
            frame_index  (int)     : 원본 프레임 인덱스
            src_filename (str)     : 원본 파일명
            corners_src  (ndarray) : 원본 4점 좌표 (4, 2), float32
            conf         (float)   : 탐지 신뢰도
            det_index    (int)     : 동일 프레임 내 탐지 순번 (복수 번호판 대응)
        Output:
            success (bool) : 저장 성공 여부
        """
        # 파일명: 파일이름_det0.png
        filename  = f"{src_filename}_det{det_index}.png"
        save_path = self.output_dir / filename


        success = cv2.imwrite(str(save_path), warped)

        log_entry = {
            "frame_index"   : frame_index,
            "det_index"     : det_index,
            "source_filename": src_filename,
            "output_filename": filename,
            "corners"       : corners_src.tolist() if corners_src is not None else None,
            "conf"          : round(conf, 4),
            "status"        : "success" if success else "save_failed",
            "timestamp"     : datetime.now().isoformat(),
        }
        self._log.append(log_entry)

        return success

    # ------------------------------------------------------------------ #
    def save_no_detection(self, frame_index: int, src_filename: str) -> None:
        """
        탐지 결과 없는 프레임을 로그에 기록한다 (이미지 저장 없음).

        Input:
            frame_index  (int) : 원본 프레임 인덱스
            src_filename (str) : 원본 파일명
        Output: (없음)
        """
        self._log.append({
            "frame_index"    : frame_index,
            "source_filename": src_filename,
            "status"         : "no_detection",
            "timestamp"      : datetime.now().isoformat(),
        })

    # ------------------------------------------------------------------ #
    def save_align_failed(
        self, frame_index: int, src_filename: str, reason: str
    ) -> None:
        """
        투시 변환 실패 프레임을 로그에 기록한다.

        Input:
            frame_index  (int) : 원본 프레임 인덱스
            src_filename (str) : 원본 파일명
            reason       (str) : 실패 사유
        Output: (없음)
        """
        self._log.append({
            "frame_index"    : frame_index,
            "source_filename": src_filename,
            "status"         : "align_failed",
            "reason"         : reason,
            "timestamp"      : datetime.now().isoformat(),
        })

    # ------------------------------------------------------------------ #
    def flush_log(self, log_path: str = "result_log.json") -> None:
        """
        누적된 로그를 JSON 파일로 저장한다.

        Input:
            log_path (str) : 저장할 JSON 경로
        Output: (없음, 파일 저장)
        """
        data = {
            "total"   : len(self._log),
            "success" : sum(1 for e in self._log if e.get("status") == "success"),
            "no_detection": sum(1 for e in self._log if e.get("status") == "no_detection"),
            "failed"  : sum(1 for e in self._log if e.get("status") not in ("success", "no_detection")),
            "results" : self._log,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[ResultExporter] 로그 저장 → {log_path}")

    # ------------------------------------------------------------------ #
    def cleanup_temp(self, temp_dir: str = "temp") -> None:
        """
        처리 완료 후 temp/ 폴더를 삭제한다.

        Input:
            temp_dir (str) : 삭제할 임시 폴더 경로
        Output: (없음)
        """
        temp_path = Path(temp_dir)
        if temp_path.exists():
            shutil.rmtree(temp_path)
            print(f"[ResultExporter] temp 폴더 삭제 완료: {temp_dir}")


# ====================================================================== #
if __name__ == "__main__":
    import sys

    out_dir  = sys.argv[1] if len(sys.argv) > 1 else "output/test"
    exporter = ResultExporter(output_dir=out_dir, prefix="aligned_frame_")

    # 더미 이미지로 save 테스트
    dummy_img     = np.zeros((170, 335, 3), dtype=np.uint8)
    dummy_corners = np.array([[10,20],[300,18],[305,160],[8,162]], dtype=np.float32)

    ok = exporter.save(
        warped       = dummy_img,
        frame_index  = 0,
        src_filename = "test.jpg",
        corners_src  = dummy_corners,
        conf         = 0.95,
        det_index    = 0,
    )
    assert ok, "이미지 저장 실패"

    exporter.save_no_detection(frame_index=1, src_filename="test2.jpg")

    exporter.flush_log(log_path=f"{out_dir}/result_log.json")

    print(f"✅ ResultExporter 단독 테스트 통과")
    print(f"   저장 경로: {out_dir}/")
