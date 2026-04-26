const placeholderPool = Array.from({ length: 5 }, (_, index) =>
  new URL(`../assets/placeholder${index + 1}.png`, import.meta.url).href,
);

/**
 * Input: {string[]} sources
 * Output: {string[]}
 * Purpose: 5개 슬롯 각각에 사용할 placeholder 이미지를 무작위로 선택한다.
 */
const createStageImageSet = (sources) =>
  Array.from({ length: 5 }, () => {
    const randomIndex = Math.floor(Math.random() * sources.length);
    return sources[randomIndex];
  });

export const USE_DUMMY = true;

export const DUMMY_RESPONSE = {
  yolo_crops: [...createStageImageSet(placeholderPool)],
  yolo_boxes: [
    { x: 120, y: 45, w: 180, h: 60, confidence: 0.97 },
    { x: 114, y: 52, w: 188, h: 58, confidence: 0.95 },
    { x: 132, y: 49, w: 176, h: 63, confidence: 0.96 },
    { x: 108, y: 47, w: 184, h: 59, confidence: 0.94 },
    { x: 126, y: 54, w: 192, h: 57, confidence: 0.98 },
  ],
  denoised: [...createStageImageSet(placeholderPool)],
  sr: [...createStageImageSet(placeholderPool)],
  ocr_text: '12가 3456',
};
