import { useEffect, useRef, useState } from 'react';
import UploadZone from './components/UploadZone/UploadZone';
import ResultGrid from './components/ResultGrid/ResultGrid';
import OcrResult from './components/OcrResult/OcrResult';
import DetailModal from './components/DetailModal/DetailModal';
import { analyzeMedia } from './services/api';
import styles from './App.module.css';

const DEFAULT_OPTIONS = {
  hrWidth: 335,
  hrHeight: 170,
  srMode: 'auto',
  denoiseEnabled: true,
};

export default function App() {
  const [inputMode, setInputMode] = useState('image');
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [videoTimeRange, setVideoTimeRange] = useState({ start: 0, end: 3 });
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modalInfo, setModalInfo] = useState(null);

  const previewsRef = useRef([]);
  const videoPreviewRef = useRef(null);

  useEffect(() => {
    previewsRef.current = previews;
  }, [previews]);

  useEffect(() => {
    videoPreviewRef.current = videoPreview;
  }, [videoPreview]);

  useEffect(() => {
    return () => {
      previewsRef.current.forEach((preview) => URL.revokeObjectURL(preview));
      if (videoPreviewRef.current) {
        URL.revokeObjectURL(videoPreviewRef.current);
      }
    };
  }, []);

  const resetResults = () => {
    setResult(null);
    setError(null);
    setStatus('idle');
    setModalInfo(null);
  };

  const addImages = (newFiles) => {
    const newPreviews = newFiles.map((file) => URL.createObjectURL(file));
    setFiles((previous) => [...previous, ...newFiles]);
    setPreviews((previous) => [...previous, ...newPreviews]);
    resetResults();
  };

  const removeImage = (index) => {
    URL.revokeObjectURL(previews[index]);
    setFiles((previous) => previous.filter((_, currentIndex) => currentIndex !== index));
    setPreviews((previous) => previous.filter((_, currentIndex) => currentIndex !== index));
    resetResults();
  };

  const setVideo = (file) => {
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file));
    resetResults();
  };

  const clearVideo = () => {
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
    setVideoFile(null);
    setVideoPreview(null);
    resetResults();
  };

  const updateOption = (key, value) => {
    setOptions((previous) => ({ ...previous, [key]: value }));
    resetResults();
  };

  const handleModeChange = (mode) => {
    setInputMode(mode);
    resetResults();
  };

  const handleAnalyzeUpload = async () => {
    setStatus('loading');
    setError(null);
    setResult(null);

    const selectedFiles = inputMode === 'image' ? files : [videoFile];

    try {
      const payload = await analyzeMedia({
        files: selectedFiles,
        inputMode,
        hrWidth: options.hrWidth,
        hrHeight: options.hrHeight,
        videoStart: videoTimeRange.start,
        videoEnd: videoTimeRange.end,
        srMode: options.srMode,
        denoiseEnabled: options.denoiseEnabled,
      });
      setResult(payload);
      setStatus('done');
    } catch (requestError) {
      setError(requestError.message);
      setStatus('error');
    }
  };

  const canAnalyze = inputMode === 'image' ? files.length > 0 : videoFile !== null;
  const isLoading = status === 'loading';

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <header className={styles.header}>
          <p className={styles.eyebrow}>AI LICENSE PLATE DEMO</p>
          <h1 className={styles.title}>Multi-Frame Target Inspector</h1>
          <p className={styles.description}>
            이미지 또는 동영상에서 번호판 후보를 추출하고 전처리, SR, OCR 단계 결과를 확인합니다.
          </p>
        </header>

        <main className={styles.content}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <h2 className={styles.panelTitle}>Upload Media</h2>
                <div className={styles.modeGroup} aria-label="input mode">
                  <label className={styles.modeOption}>
                    <input
                      type="radio"
                      name="mode"
                      checked={inputMode === 'image'}
                      onChange={() => handleModeChange('image')}
                    />
                    Image
                  </label>
                  <label className={styles.modeOption}>
                    <input
                      type="radio"
                      name="mode"
                      checked={inputMode === 'video'}
                      onChange={() => handleModeChange('video')}
                    />
                    Video
                  </label>
                </div>
              </div>
              <button
                type="button"
                className={styles.analyzeButton}
                disabled={!canAnalyze || isLoading}
                onClick={handleAnalyzeUpload}
              >
                {isLoading ? 'Processing...' : 'Analyze'}
              </button>
            </div>

            <div className={styles.controlsGrid}>
              <label className={styles.field}>
                <span>HR Width</span>
                <input
                  type="number"
                  min="1"
                  value={options.hrWidth}
                  onChange={(event) => updateOption('hrWidth', Number(event.target.value))}
                />
              </label>
              <label className={styles.field}>
                <span>HR Height</span>
                <input
                  type="number"
                  min="1"
                  value={options.hrHeight}
                  onChange={(event) => updateOption('hrHeight', Number(event.target.value))}
                />
              </label>
              <label className={styles.field}>
                <span>SR Mode</span>
                <select
                  value={options.srMode}
                  onChange={(event) => updateOption('srMode', event.target.value)}
                >
                  <option value="auto">Auto</option>
                  <option value="always">Always</option>
                  <option value="skip">Skip</option>
                </select>
              </label>
              <label className={styles.toggleField}>
                <input
                  type="checkbox"
                  checked={options.denoiseEnabled}
                  onChange={(event) => updateOption('denoiseEnabled', event.target.checked)}
                />
                Denoise
              </label>
            </div>

            <UploadZone
              inputMode={inputMode}
              files={files}
              previews={previews}
              videoFile={videoFile}
              videoPreview={videoPreview}
              onAddImages={addImages}
              onRemoveImage={removeImage}
              onSetVideo={setVideo}
              onClearVideo={clearVideo}
              onTimeRangeChange={(start, end) => setVideoTimeRange({ start, end })}
            />

            {error ? <p className={styles.errorBanner}>{error}</p> : null}
          </section>

          {result ? (
            <>
              <section className={styles.summaryBar}>
                <span>Route: {result.pipeline_route}</span>
                <span>SR: {result.sr_applied ? 'Applied' : 'Skipped'}</span>
                <span>High-res crops: {result.high_resolution_count}</span>
                <span>
                  HR: {result.hr_width} x {result.hr_height}
                </span>
              </section>
              <ResultGrid
                result={result}
                previews={previews}
                onDetailOpen={(info) => setModalInfo(info)}
              />
              <OcrResult text={result.ocr_text} />
            </>
          ) : null}
        </main>
      </div>

      {isLoading ? (
        <div className={styles.loadingOverlay}>
          <div className={styles.spinner} />
          <p className={styles.loadingText}>Analyzing media...</p>
        </div>
      ) : null}

      <DetailModal
        modalInfo={modalInfo}
        result={result}
        previews={previews}
        onClose={() => setModalInfo(null)}
      />
    </div>
  );
}
