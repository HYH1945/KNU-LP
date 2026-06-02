import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import styles from './DetailModal.module.css';

const getInputImage = (result, previews, index) =>
  result?.input_preview?.[index] ?? result?.selected_inputs?.[index] ?? previews[index];

const formatPlateBbox = (result, index) => {
  const bbox = result?.selected_plate_bboxes?.[index];
  if (!bbox) {
    return null;
  }
  if (!bbox.detected) {
    return 'Original plate bbox: not detected';
  }

  const confidence =
    typeof bbox.confidence === 'number' ? `, conf ${(bbox.confidence * 100).toFixed(1)}%` : '';
  return `Original plate bbox: ${bbox.width} x ${bbox.height} (area ${bbox.area}${confidence})`;
};

function ImageWithDimensions({ className, src, alt, details = [] }) {
  const [dimensions, setDimensions] = useState(null);

  useEffect(() => {
    setDimensions(null);
  }, [src]);

  return (
    <div className={styles.imageBlock}>
      <img
        className={className}
        src={src}
        alt={alt}
        onLoad={(event) => {
          setDimensions({
            width: event.currentTarget.naturalWidth,
            height: event.currentTarget.naturalHeight,
          });
        }}
      />
      <p className={styles.dimensionText}>
        {dimensions ? `${dimensions.width} x ${dimensions.height}` : 'Loading size...'}
      </p>
      {details.filter(Boolean).map((detail) => (
        <p key={detail} className={styles.detailText}>
          {detail}
        </p>
      ))}
    </div>
  );
}

const getModalContent = (stage, index, result, previews) => {
  if (!result && stage !== 'INPUT') {
    return null;
  }

  if (stage === 'INPUT') {
    return {
      title: `INPUT ${index + 1}`,
      content: (
        <ImageWithDimensions
          className={styles.singleImage}
          src={getInputImage(result, previews, index)}
          alt={`input-detail-${index + 1}`}
        />
      ),
    };
  }

  if (stage === 'YOLO') {
    const plateBbox = formatPlateBbox(result, index);

    return {
      title: `YOLO Crop ${index + 1}`,
      content: (
        <div className={styles.compareGrid}>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>Selected</p>
            <ImageWithDimensions
              className={styles.compareImage}
              src={result.yolo_selected[index]}
              alt={`selected-compare-${index + 1}`}
              details={[plateBbox]}
            />
          </div>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>Crop</p>
            <ImageWithDimensions
              className={styles.compareImage}
              src={result.yolo_crops[index]}
              alt={`crop-compare-${index + 1}`}
              details={[plateBbox]}
            />
          </div>
        </div>
      ),
    };
  }

  if (stage === 'Denoised') {
    const plateBbox = formatPlateBbox(result, index);

    return {
      title: `Denoised ${index + 1}`,
      content: (
        <div className={styles.compareGrid}>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>Crop</p>
            <ImageWithDimensions
              className={styles.compareImage}
              src={result.yolo_crops[index]}
              alt={`crop-denoise-${index + 1}`}
              details={[plateBbox]}
            />
          </div>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>Denoised</p>
            <ImageWithDimensions
              className={styles.compareImage}
              src={result.denoised[index]}
              alt={`denoised-detail-${index + 1}`}
              details={[plateBbox]}
            />
          </div>
        </div>
      ),
    };
  }

  if (stage === 'SR') {
    const plateBbox = formatPlateBbox(result, index);

    return {
      title: `SR ${index + 1}`,
      content: (
        <div className={styles.compareGrid}>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>Denoised</p>
            <ImageWithDimensions
              className={styles.compareImage}
              src={result.denoised[index]}
              alt={`lr-detail-${index + 1}`}
              details={[plateBbox]}
            />
          </div>
          <div className={styles.compareCard}>
            <p className={styles.compareLabel}>SR</p>
            <ImageWithDimensions
              className={styles.compareImage}
              src={result.sr[index]}
              alt={`sr-detail-${index + 1}`}
              details={[plateBbox]}
            />
          </div>
        </div>
      ),
    };
  }

  return null;
};

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
      modalInfo.triggerRef?.current?.focus();
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
            x
          </button>
        </div>
        <div className={styles.body}>{modalContent.content}</div>
      </div>
    </div>,
    document.body,
  );
}
