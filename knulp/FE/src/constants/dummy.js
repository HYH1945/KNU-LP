const placeholderPool = Array.from({ length: 5 }, (_, index) =>
  new URL(`../assets/placeholder${index + 1}.png`, import.meta.url).href,
);

const createStageImageSet = (sources) =>
  Array.from({ length: 5 }, () => {
    const randomIndex = Math.floor(Math.random() * sources.length);
    return sources[randomIndex];
  });

export const USE_DUMMY = false;

export const DUMMY_RESPONSE = {
  input_preview: [...createStageImageSet(placeholderPool)],
  input_omitted_count: 7,
  selected_inputs: [...createStageImageSet(placeholderPool)],
  selected_source_indices: [0, 2, 3, 4, 6],
  selected_plate_bboxes: [
    { detected: true, width: 246, height: 74, area: 18204, confidence: 0.93, source_index: 0 },
    { detected: true, width: 231, height: 69, area: 15939, confidence: 0.9, source_index: 2 },
    { detected: true, width: 220, height: 66, area: 14520, confidence: 0.87, source_index: 3 },
    { detected: true, width: 205, height: 61, area: 12505, confidence: 0.84, source_index: 4 },
    { detected: true, width: 190, height: 58, area: 11020, confidence: 0.8, source_index: 6 },
  ],
  yolo_crops: [...createStageImageSet(placeholderPool)],
  yolo_selected: [...createStageImageSet(placeholderPool)],
  denoised: [...createStageImageSet(placeholderPool)],
  sr: [...createStageImageSet(placeholderPool)],
  ocr_text: '12가 3456',
  pipeline_route: 'sr_then_ocr',
  sr_applied: true,
  high_resolution_count: 0,
  hr_width: 96,
  hr_height: 32,
};
