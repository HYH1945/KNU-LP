"""
main.py
- plate_detection_seg 전체 파이프라인 진입점
- ImageLoader → PlateSegmenter → PerspectiveAligner → ResultExporter 순서 실행
- 이미지셋 폴더 자동 감지, 중단 재개 지원
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

from image_loader        import ImageLoader
from plate_segmenter     import PlateSegmenter
from perspective_aligner import PerspectiveAligner
from result_exporter     import ResultExporter
from progress_manager    import ProgressManager


# ====================================================================== #
def load_config(path: str = "configs/settings.yaml") -> dict:
    """
    Input:
        path (str) : settings.yaml 경로
    Output:
        config (dict) : 설정 딕셔너리
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ====================================================================== #
def get_imagesets(input_root: str) -> list[Path]:
    """
    input/ 하위의 이미지셋 폴더 목록을 반환한다.

    Input:
        input_root (str) : input 루트 경로
    Output:
        imagesets (list[Path]) : 이미지셋 폴더 경로 리스트
    """
    root = Path(input_root)
    imagesets = sorted([p for p in root.iterdir() if p.is_dir()])
    if not imagesets:
        raise FileNotFoundError(f"이미지셋 폴더 없음: {input_root}")
    return imagesets


# ====================================================================== #
def process_imageset(
    imageset_dir: Path,
    output_dir  : Path,
    segmenter   : PlateSegmenter,
    aligner     : PerspectiveAligner,
    cfg         : dict,
) -> None:
    """
    단일 이미지셋을 처리한다.

    Input:
        imageset_dir (Path)              : 처리할 이미지셋 폴더
        output_dir   (Path)              : 결과 저장 폴더
        segmenter    (PlateSegmenter)    : 탐지 모듈
        aligner      (PerspectiveAligner): 투시 변환 모듈
        cfg          (dict)              : settings.yaml 설정값
    Output: (없음, 파일 저장)
    """
    imageset_name = imageset_dir.name
    print(f"\n{'='*55}")
    print(f"  이미지셋: {imageset_name}")
    print(f"{'='*55}")

    # ── 모듈 초기화 ────────────────────────────────────────────────── #
    loader   = ImageLoader(str(imageset_dir))
    exporter = ResultExporter(
        output_dir = str(output_dir),
        prefix     = cfg["output"]["prefix"],
    )
    progress = ProgressManager(cfg["progress"]["file"])

    total      = loader.total
    start_idx  = progress.load(imageset_name, total)

    if start_idx >= total:
        print(f"  이미 완료된 이미지셋, 건너뜀")
        return

    print(f"  총 {total}장 | {start_idx}번부터 시작\n")

    # ── 프레임별 처리 ──────────────────────────────────────────────── #
    success_count = 0
    for idx, frame, path in loader.iter_frames(start_index=start_idx):

        print(f"  [{idx+1}/{total}] {path.name}", end=" → ")

        # Phase 2.1: 번호판 탐지
        detections = segmenter.detect(frame)

        if not detections:
            print("탐지 없음")
            exporter.save_no_detection(idx, path.name)
            progress.update(idx)
            continue

        print(f"{len(detections)}개 탐지")

        # Phase 2.2: 투시 보정 (탐지된 모든 번호판 처리)
        for det_i, det in enumerate(detections):
            result = aligner.align(frame, det)

            if not result.success:
                print(f"    └─ det{det_i}: 보정 실패 ({result.reason})")
                exporter.save_align_failed(idx, path.name, result.reason)
                continue

            if(det.conf < cfg["model"]["conf_thr"]):
                print(f"    └─ det{det_i}: 신뢰도 낮음 (conf={det.conf:.2f})")
                exporter.save_align_failed(idx, path.name, f"low_confidence_{det.conf:.2f}")
                continue

            ok = exporter.save(
                warped       = result.warped,
                frame_index  = idx,
                src_filename = path.name,
                corners_src  = result.corners_src,
                conf         = det.conf,
                det_index    = det_i,
            )
            status = "저장 완료" if ok else "저장 실패"
            print(f"    └─ det{det_i}: conf={det.conf:.2f} | {status}")
            if ok:
                success_count += 1

        progress.update(idx)

    # ── 완료 처리 ──────────────────────────────────────────────────── #
    log_path = output_dir / "result_log.json"
    exporter.flush_log(str(log_path))
    progress.complete()
    exporter.cleanup_temp(cfg["progress"]["file"].split("/")[0])

    print(f"\n  완료: {success_count}장 저장 → {output_dir}")


# ====================================================================== #
def parse_args() -> argparse.Namespace:
    """
    Input : sys.argv
    Output: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="번호판 투시 보정 파이프라인")
    parser.add_argument("--config",   default="configs/settings.yaml", help="설정 파일 경로")
    parser.add_argument("--imageset", default=None, help="특정 이미지셋 이름 지정 (없으면 전체)")
    parser.add_argument("--conf",     type=float, default=None, help="탐지 신뢰도 임계값 오버라이드")
    return parser.parse_args()


# ====================================================================== #
def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI 오버라이드
    if args.conf:
        cfg["model"]["conf_thr"] = args.conf

    # ── 모델 초기화 (이미지셋 전체에서 재사용) ─────────────────────── #
    segmenter = PlateSegmenter(
        weights  = cfg["paths"]["weights"],
        device   = cfg["model"]["device"],
        conf_thr = cfg["model"]["conf_thr"],
        iou_thr  = cfg["model"]["iou_thr"],
        imgsz    = cfg["model"]["imgsz"],
    )
    segmenter.warmup()

    aligner = PerspectiveAligner(
        output_width  = cfg["plate"]["output_width"],
        output_height = cfg["plate"]["output_height"],
    )

    # ── 이미지셋 목록 ──────────────────────────────────────────────── #
    imagesets = get_imagesets(cfg["paths"]["input_root"])

    if args.imageset:
        imagesets = [p for p in imagesets if p.name == args.imageset]
        if not imagesets:
            print(f"이미지셋 없음: {args.imageset}")
            sys.exit(1)

    print(f"\n처리할 이미지셋: {[p.name for p in imagesets]}")

    # ── 이미지셋별 처리 ────────────────────────────────────────────── #
    for imageset_dir in imagesets:
        output_dir = Path(cfg["paths"]["output_root"]) / imageset_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        process_imageset(
            imageset_dir = imageset_dir,
            output_dir   = output_dir,
            segmenter    = segmenter,
            aligner      = aligner,
            cfg          = cfg,
        )

    print("\n✅ 전체 파이프라인 완료")


# ====================================================================== #
if __name__ == "__main__":
    main()
