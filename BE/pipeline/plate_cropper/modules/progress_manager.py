"""
ProgressManager
- temp/progress.json을 통해 처리 상태를 기록하고
  프로그램 중단 시 마지막 처리 인덱스 이후부터 재개하는 상태 관리 모듈
"""

import json
import os
from pathlib import Path
from datetime import datetime


class ProgressManager:
    """
    파이프라인 처리 상태를 JSON 파일로 관리.
    """

    def __init__(self, progress_file: str = "temp/progress.json"):
        """
        Input:
            progress_file (str) : 상태 파일 경로
        """
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict = {}

    # ------------------------------------------------------------------ #
    def load(self, imageset_name: str, total: int) -> int:
        """
        저장된 상태를 로드하고 시작 인덱스를 반환한다.

        Input:
            imageset_name (str) : 현재 처리 중인 이미지셋 이름
            total         (int) : 전체 이미지 수
        Output:
            start_index (int) : 재개할 인덱스 (처음이면 0)
        """
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                self._state = json.load(f)

            # 같은 이미지셋이면 이어서
            if self._state.get("imageset") == imageset_name:
                last = self._state.get("last_processed_index", -1)
                start = last + 1
                if start < total:
                    print(f"[ProgressManager] 재개: {imageset_name} | {start}/{total} 부터")
                    return start
                else:
                    print(f"[ProgressManager] 이미 완료된 이미지셋: {imageset_name}")
                    return total   # 전부 완료됨

        # 신규 시작
        self._state = {
            "imageset"            : imageset_name,
            "total"               : total,
            "last_processed_index": -1,
            "status"              : "in_progress",
            "started_at"          : datetime.now().isoformat(),
        }
        self._write()
        print(f"[ProgressManager] 신규 시작: {imageset_name} | 총 {total}장")
        return 0

    # ------------------------------------------------------------------ #
    def update(self, index: int) -> None:
        """
        현재 처리 인덱스를 기록한다.

        Input:
            index (int) : 방금 처리 완료된 프레임 인덱스
        Output: (없음, 파일 갱신)
        """
        self._state["last_processed_index"] = index
        self._state["updated_at"]           = datetime.now().isoformat()
        self._write()

    # ------------------------------------------------------------------ #
    def complete(self) -> None:
        """
        처리 완료 상태로 표시한다.

        Input : (없음)
        Output: (없음, 파일 갱신)
        """
        self._state["status"]       = "completed"
        self._state["completed_at"] = datetime.now().isoformat()
        self._write()
        print("[ProgressManager] 완료 상태 기록")

    # ------------------------------------------------------------------ #
    def delete(self) -> None:
        """
        상태 파일을 삭제한다 (ResultExporter.cleanup_temp 전에 호출).

        Input : (없음)
        Output: (없음)
        """
        if self.progress_file.exists():
            self.progress_file.unlink()
            print("[ProgressManager] 상태 파일 삭제")

    # ------------------------------------------------------------------ #
    def _write(self) -> None:
        """
        Input : (없음, self._state 사용)
        Output: (없음, 파일 저장)
        """
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)


# ====================================================================== #
if __name__ == "__main__":
    import shutil

    pm = ProgressManager("temp/test_progress.json")

    start = pm.load("imageset_1", total=100)
    assert start == 0, "신규 시작 인덱스 오류"

    pm.update(49)
    pm.update(50)

    # 재로드 테스트
    pm2    = ProgressManager("temp/test_progress.json")
    start2 = pm2.load("imageset_1", total=100)
    assert start2 == 51, f"재개 인덱스 오류: {start2}"

    pm2.complete()
    pm2.delete()

    shutil.rmtree("temp", ignore_errors=True)
    print("✅ ProgressManager 단독 테스트 통과")
