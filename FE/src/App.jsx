import { useEffect, useRef, useState } from 'react';
import UploadZone from './components/UploadZone/UploadZone';
import ResultGrid from './components/ResultGrid/ResultGrid';
import OcrResult from './components/OcrResult/OcrResult';
import DetailModal from './components/DetailModal/DetailModal';
import { analyzeImages } from './services/api';
import styles from './App.module.css';

const SLOT_COUNT = 5;

/**
 * Input: 없음.
 * Output: {(File|null)[]}
 * Purpose: 5개 고정 길이의 빈 파일 슬롯 배열을 생성한다.
 */
const createEmptySlots = () => Array.from({ length: SLOT_COUNT }, () => null);

/**
 * Input: {(string|null)[]} targetPreviews
 * Output: {void}
 * Purpose: previews 배열에 남아 있는 모든 Object URL을 해제한다.
 */
const revokePreviewUrls = (targetPreviews) => {
  targetPreviews.forEach((preview) => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
  });
};

/**
 * Input: 없음.
 * Output: {JSX.Element}
 * Purpose: 번호판 인식 데모 프런트엔드의 전체 상태와 화면을 관리한다.
 */
export default function App() {
  const [files, setFiles] = useState(createEmptySlots);
  const [previews, setPreviews] = useState(createEmptySlots);
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modalInfo, setModalInfo] = useState(null);
  const latestPreviewsRef = useRef(previews);

  useEffect(() => {
    latestPreviewsRef.current = previews;
  }, [previews]);

  useEffect(() => () => {
    revokePreviewUrls(latestPreviewsRef.current);
  }, []);

  /**
   * Input: {number} index, {File} file
   * Output: {void}
   * Purpose: 지정 슬롯의 파일과 preview URL을 교체하고 기존 URL은 즉시 해제한다.
   */
  const setSlot = (index, file) => {
    setPreviews((currentPreviews) => {
      const nextPreviews = [...currentPreviews];

      if (nextPreviews[index]) {
        URL.revokeObjectURL(nextPreviews[index]);
      }

      nextPreviews[index] = URL.createObjectURL(file);
      latestPreviewsRef.current = nextPreviews;
      return nextPreviews;
    });

    setFiles((currentFiles) => {
      const nextFiles = [...currentFiles];
      nextFiles[index] = file;
      return nextFiles;
    });

    setResult(null);
    setError(null);
    setStatus('idle');
    setModalInfo((currentModal) => (currentModal?.index === index ? null : currentModal));
  };

  /**
   * Input: {number} index
   * Output: {void}
   * Purpose: 지정 슬롯의 파일과 preview URL을 제거하고 결과 상태를 초기화한다.
   */
  const clearSlot = (index) => {
    setPreviews((currentPreviews) => {
      const nextPreviews = [...currentPreviews];

      if (nextPreviews[index]) {
        URL.revokeObjectURL(nextPreviews[index]);
      }

      nextPreviews[index] = null;
      latestPreviewsRef.current = nextPreviews;
      return nextPreviews;
    });

    setFiles((currentFiles) => {
      const nextFiles = [...currentFiles];
      nextFiles[index] = null;
      return nextFiles;
    });

    setResult(null);
    setError(null);
    setStatus('idle');
    setModalInfo((currentModal) => (currentModal?.index === index ? null : currentModal));
  };

  /**
   * Input: 없음.
   * Output: {Promise<void>}
   * Purpose: 업로드된 5장 이미지를 분석하고 결과 상태를 반영한다.
   */
  const handleAnalyze = async () => {
    setStatus('loading');
    setError(null);

    try {
      const response = await analyzeImages(files);
      setResult(response);
      setStatus('done');
    } catch (requestError) {
      setResult(null);
      setStatus('error');
      setError(requestError instanceof Error ? requestError.message : 'unknown error');
    }
  };

  /**
   * Input: {{stage: string, index: number, triggerRef: { current: HTMLElement|null }}} info
   * Output: {void}
   * Purpose: 상세 보기 모달에 표시할 대상 정보를 저장한다.
   */
  const handleDetailOpen = (info) => {
    setModalInfo(info);
  };

  /**
   * Input: 없음.
   * Output: {void}
   * Purpose: 상세 보기 모달을 닫고 마지막 트리거 버튼으로 포커스를 복귀시킨다.
   */
  const handleDetailClose = () => {
    setModalInfo((currentModal) => {
      currentModal?.triggerRef?.current?.focus();
      return null;
    });
  };

  const canAnalyze = files.every((file) => file !== null) && status !== 'loading';

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <header className={styles.header}>
          <p className={styles.eyebrow}>AI LICENSE PLATE DEMO</p>
          <h1 className={styles.title}>5-image Pipeline Inspector</h1>
          <p className={styles.description}>
            입력 5장을 기준으로 YOLO Crop, Denoised, SR 단계를 한 화면에서 비교하고
            OCR 결과를 확인할 수 있습니다.
          </p>
        </header>

        <main className={styles.content}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <h2 className={styles.panelTitle}>Upload</h2>
                <p className={styles.panelText}>
                  한 번에 여러 장을 올리면 빈 슬롯부터 채워지고, 각 슬롯은 개별 교체와
                  삭제를 지원합니다.
                </p>
              </div>
              <button
                type="button"
                className={styles.analyzeButton}
                disabled={!canAnalyze}
                onClick={handleAnalyze}
              >
                {status === 'loading' ? '분석 중...' : '분석 시작'}
              </button>
            </div>

            <UploadZone
              files={files}
              previews={previews}
              onSlotChange={setSlot}
              onSlotClear={clearSlot}
            />

            {status === 'error' ? <p className={styles.errorBanner}>{error}</p> : null}
          </section>

          {result ? (
            <>
              <ResultGrid
                result={result}
                previews={previews}
                onDetailOpen={handleDetailOpen}
              />
              <OcrResult text={result.ocr_text} />
            </>
          ) : null}
        </main>
      </div>

      <DetailModal
        modalInfo={modalInfo}
        result={result}
        previews={previews}
        onClose={handleDetailClose}
      />

      {status === 'loading' ? (
        <div className={styles.loadingOverlay} role="status" aria-live="polite">
          <div className={styles.spinner} />
          <p className={styles.loadingText}>5장 이미지를 분석하고 있습니다.</p>
        </div>
      ) : null}
    </div>
  );
}
