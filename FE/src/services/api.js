import { DUMMY_RESPONSE, USE_DUMMY } from '../constants/dummy';

/**
 * Input: {unknown} response
 * Output: {void}
 * Purpose: 파이프라인 응답의 5개 고정 길이 배열 스키마를 검증한다.
 */
const validateResponseShape = (response) => {
  const targetKeys = ['yolo_crops', 'yolo_boxes', 'denoised', 'sr'];
  const isValid = targetKeys.every(
    (key) => Array.isArray(response?.[key]) && response[key].length === 5,
  );

  if (!isValid) {
    throw new Error('invalid response shape');
  }
};

/**
 * Input: {(File|null)[]} files
 * Output: {Promise<object>}
 * Purpose: 더미 모드 또는 실서버 모드에서 번호판 분석 결과를 반환한다.
 */
export const analyzeImages = async (files) => {
  if (USE_DUMMY) {
    const dummyResponse = await new Promise((resolve) => {
      window.setTimeout(() => resolve(DUMMY_RESPONSE), 800);
    });

    validateResponseShape(dummyResponse);
    return dummyResponse;
  }

  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await fetch('/api/analyze', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`request failed: ${response.status}`);
  }

  const payload = await response.json();
  validateResponseShape(payload);
  return payload;
};
