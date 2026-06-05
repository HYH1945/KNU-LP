import { DUMMY_RESPONSE, USE_DUMMY } from '../constants/dummy';

const validateResponseShape = (response) => {
  const targetKeys = ['input_preview', 'selected_inputs', 'yolo_crops', 'yolo_selected', 'denoised'];
  const isValid = targetKeys.every(
    (key) => Array.isArray(response?.[key]) && response[key].length === 5,
  );
  const hasSrImages =
    Array.isArray(response?.sr) && response.sr.length >= 1 && response.sr.length <= 5;
  const hasBboxMetadata =
    Array.isArray(response?.selected_plate_bboxes) &&
    response.selected_plate_bboxes.length === 5;

  if (
    !isValid ||
    !hasSrImages ||
    !hasBboxMetadata ||
    typeof response?.ocr_text !== 'string' ||
    typeof response?.input_omitted_count !== 'number'
  ) {
    throw new Error('invalid response shape');
  }
};

export const analyzeMedia = async ({
  files,
  inputMode,
  hrWidth,
  hrHeight,
  videoStart,
  videoEnd,
  srMode,
  denoiseEnabled,
}) => {
  if (USE_DUMMY) {
    const dummyResponse = await new Promise((resolve) => {
      window.setTimeout(() => resolve(DUMMY_RESPONSE), 800);
    });

    validateResponseShape(dummyResponse);
    return dummyResponse;
  }

  const formData = new FormData();
  files.filter(Boolean).forEach((file) => {
    formData.append('files', file);
  });
  formData.append('input_mode', inputMode);
  formData.append('hr_width', String(hrWidth));
  formData.append('hr_height', String(hrHeight));
  formData.append('sr_mode', srMode);
  formData.append('denoise_enabled', String(denoiseEnabled));

  if (inputMode === 'video') {
    formData.append('video_start', String(videoStart));
    formData.append('video_end', String(videoEnd));
  }

  const response = await fetch('/api/analyze', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `request failed: ${response.status}`);
  }

  const payload = await response.json();
  validateResponseShape(payload);
  return payload;
};

export const analyzeImages = async (files) =>
  analyzeMedia({
    files,
    inputMode: 'image',
    hrWidth: 96,
    hrHeight: 32,
    videoStart: 0,
    videoEnd: 3,
    srMode: 'auto',
    denoiseEnabled: true,
  });
