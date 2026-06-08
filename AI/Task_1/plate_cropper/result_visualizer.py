import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import yaml

from modules.plate_segmenter import PlateSegmenter
from modules.perspective_aligner import PerspectiveAligner

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        mask = det.mask_full
        color_mask = np.zeros_like(annotated)
        color_mask[mask > 0] = (0, 0, 255)
        
        alpha = 0.5
        mask_indices = mask > 0
        annotated[mask_indices] = cv2.addWeighted(
            annotated[mask_indices], 1 - alpha, 
            color_mask[mask_indices], alpha, 0
        )
    return annotated

def vconcat_crops(crops, target_width=300):
    if not crops:
        return None
    if not isinstance(crops, list):
        crops = [crops]
        
    resized_crops = []
    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        if w == 0: continue
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)
        if target_height > 0:
            resized = cv2.resize(crop, (target_width, target_height))
            resized_crops.append(resized)
            
    if not resized_crops:
        return None
        
    return cv2.vconcat(resized_crops)

def main():
    config_path = Path(r"C:\plate_project\code\plate_detection_seg\configs\settings.yaml")
    if not config_path.exists():
        print(f"설정 파일 없음: {config_path}")
        sys.exit(1)
        
    cfg = load_config(config_path)

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

    input_dir = Path(cfg["paths"]["input_root"]) / "visualize"
    output_dir = Path(cfg["paths"]["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.glob(ext))
    
    max_images = 5
    image_paths = sorted(image_paths)[:max_images]
    num_images = len(image_paths)

    if num_images == 0:
        print(f"처리할 이미지 없음: {input_dir}")
        sys.exit(1)

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)

    cols = ['Original Image', 'Segmented Image', 'Aligned Image(s)']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=16, fontweight='bold')

    for i, img_path in enumerate(image_paths):
        bgr_img = cv2.imread(str(img_path))
        if bgr_img is None: continue
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        detections = segmenter.detect(bgr_img)

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].axis('off')

        annotated_bgr = draw_detections(bgr_img, detections)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        axes[i, 1].imshow(annotated_rgb)
        axes[i, 1].axis('off')

        aligned_crops_bgr = []
        for det in detections:
            align_result = aligner.align(bgr_img, det)
            if align_result is not None and hasattr(align_result, 'warped'):
                if align_result.warped is not None:
                    aligned_crops_bgr.append(align_result.warped)

        concatenated_crops_bgr = vconcat_crops(aligned_crops_bgr)
        
        if concatenated_crops_bgr is not None:
            concatenated_crops_rgb = cv2.cvtColor(concatenated_crops_bgr, cv2.COLOR_BGR2RGB)
            axes[i, 2].imshow(concatenated_crops_rgb)
        else:
            placeholder = np.ones((300, 300, 3), dtype=np.uint8) * 240
            axes[i, 2].imshow(placeholder)
            axes[i, 2].text(150, 150, 'No Detection', ha='center', va='center', color='red', fontsize=12)
            
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_filename = output_dir / "segmentation_summary_table.png"
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"표 생성 완료: {save_filename}")

if __name__ == "__main__":
    main()