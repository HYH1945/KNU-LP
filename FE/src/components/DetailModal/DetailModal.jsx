import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import styles from './DetailModal.module.css';

/**
 * Input: {string} stage, {number} index, {object|null} result, {(string|null)[]} previews
 * Output: {{ title: string, content: JSX.Element }|null}
 * Purpose: 선택된 stage와 index에 맞는 상세 모달 본문을 구성한다.
 */
const getModalContent = (stage, index, result, previews) => {
  if (!result && stage !== 'INPUT') {
    return null;
  }

  if (stage === 'INPUT') {
    return {
      title: `INPUT ${index + 1}`,
      content: (
        <img className={styles.singleImage} src={previews[index]} alt={`input-detail-${index + 1}`} />
      ),
    };
  }

  if (stage === 'YOLO') {
    return {
      title: `YOLO Crop ${index + 1}`,
      content: (
        <div className={styles.compareGrid}>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>선택된 부분</p>
            <img
              className={styles.compareImage}
              src={result.yolo_selected[index]}
              alt={`input-compare-${index + 1}`}
            />
          </div>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>크롭된 부분</p>
            <img
              className={styles.compareImage}
              src={result.yolo_crops[index]}
              alt={`input-compare-${index + 1}`}
            />
          </div>
        </div>
      ),
    };
  }

  if (stage === 'Denoised') {
    return {
      title: `Denoised ${index + 1}`,
      content: (
        <div className={styles.compareGrid}>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>원본</p>
            <img
              className={styles.compareImage}
              src={result.yolo_crops[index]}
              alt={`input-compare-${index + 1}`}
            />
          </div>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>Denoised</p>
            <img
              className={styles.compareImage}
              src={result.denoised[index]}
              alt={`denoised-detail-${index + 1}`}
            />
          </div>
        </div>
      ),
    };
  }

  if (stage === 'SR') {
    return {
      title: `SR ${index + 1}`,
      content: (
        <div className={styles.compareGrid}>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>LR</p>
            <img
              className={styles.compareImage}
              src={result.denoised[index]}
              alt={`lr-detail-${index + 1}`}
            />
          </div>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>SR</p>
            <img className={styles.compareImage} src={result.sr[index]} alt={`sr-detail-${index + 1}`} />
          </div>
        </div>
      ),
    };
  }

  return null;
};

/**
 * Input: {{
 *   modalInfo: null|{ stage: string, index: number, triggerRef: { current: HTMLElement|null } },
 *   result: object|null,
 *   previews: (string|null)[],
 *   onClose: ()=>void
 * }}
 * Output: {JSX.Element|null}
 * Purpose: 포털 기반 상세 모달을 렌더링하고 접근성/포커스를 제어한다.
 */
export default function DetailModal({ modalInfo, result, previews, onClose }) {
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (!modalInfo) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    closeButtonRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [modalInfo, onClose]);

  if (!modalInfo) {
    return null;
  }

  const modalContent = getModalContent(modalInfo.stage, modalInfo.index, result, previews);

  if (!modalContent) {
    return null;
  }

  return createPortal(
    <div
      className={styles.backdrop}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div className={styles.dialog} role="dialog" aria-modal="true" aria-labelledby="detail-modal-title">
        <div className={styles.header}>
          <h2 id="detail-modal-title" className={styles.title}>
            {modalContent.title}
          </h2>
          <button
            ref={closeButtonRef}
            type="button"
            className={styles.closeButton}
            aria-label="close-detail-modal"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <div className={styles.body}>{modalContent.content}</div>
      </div>
    </div>,
    document.body,
  );
}
