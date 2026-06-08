import { createRef, useRef } from 'react';
import styles from './ResultGrid.module.css';

const ROWS = [
  {
    label: 'INPUT',
    stage: 'INPUT',
    getImages: ({ result, previews }) => result?.input_preview ?? result?.selected_inputs ?? previews,
  },
  { label: 'YOLO Crop', stage: 'YOLO', getImages: ({ result }) => result.yolo_crops },
  { label: 'Denoised', stage: 'Denoised', getImages: ({ result }) => result.denoised },
  { label: 'SR', stage: 'SR', getImages: ({ result }) => result.sr },
];

const createTriggerRefs = (count) => Array.from({ length: count }, () => createRef());

const normalizeSlots = (images) => Array.from({ length: 5 }, (_, index) => images?.[index] ?? null);

export default function ResultGrid({ result, previews, onDetailOpen }) {
  const triggerRefs = useRef(
    ROWS.reduce((accumulator, row) => {
      accumulator[row.stage] = createTriggerRefs(5);
      return accumulator;
    }, {}),
  );

  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <h2 className={styles.title}>Pipeline Grid</h2>
        <p className={styles.text}>입력과 각 처리 단계를 5개 슬롯 기준으로 비교합니다.</p>
      </div>

      <div className={styles.rows}>
        {ROWS.map((row) => {
          const images = normalizeSlots(row.getImages({ result, previews }));
          const omittedCount =
            row.stage === 'INPUT'
              ? result?.input_omitted_count ?? Math.max(0, (previews?.length ?? 0) - 5)
              : 0;

          return (
            <div key={row.stage} className={styles.row}>
              <div className={styles.label}>{row.label}</div>
              <div className={styles.cells}>
                {images.map((image, index) => {
                  const buttonRef = triggerRefs.current[row.stage][index];

                  return (
                    <div key={`${row.stage}-${index}`} className={styles.cell}>
                      {image ? (
                        <img
                          className={styles.image}
                          src={image}
                          alt={`${row.label}-${index + 1}`}
                        />
                      ) : (
                        <div className={styles.empty}>No image</div>
                      )}
                      {omittedCount > 0 && index === images.length - 1 ? (
                        <span className={styles.omittedBadge}>+{omittedCount}</span>
                      ) : null}
                      {image ? (
                        <div className={styles.overlay}>
                          <button
                            ref={buttonRef}
                            type="button"
                            className={styles.detailButton}
                            onClick={() =>
                              onDetailOpen({
                                stage: row.stage,
                                index,
                                triggerRef: buttonRef,
                              })
                            }
                          >
                            Detail
                          </button>
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
