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

export const USE_DUMMY = false;

export const DUMMY_RESPONSE = {
  yolo_crops: [...createStageImageSet(placeholderPool)],
  yolo_selected: [...createStageImageSet(placeholderPool)],
  denoised: [...createStageImageSet(placeholderPool)],
  sr: [...createStageImageSet(placeholderPool)],
  ocr_text: '12가 3456',
};
