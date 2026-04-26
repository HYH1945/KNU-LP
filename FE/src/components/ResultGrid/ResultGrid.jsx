import { createRef, useRef } from 'react';
import styles from './ResultGrid.module.css';

const ROWS = [
  { label: 'INPUT', stage: 'INPUT', getImages: ({ previews }) => previews },
  { label: 'YOLO Crop', stage: 'YOLO', getImages: ({ result }) => result.yolo_crops },
  { label: 'Denoised', stage: 'Denoised', getImages: ({ result }) => result.denoised },
  { label: 'SR', stage: 'SR', getImages: ({ result }) => result.sr },
];

/**
 * Input: {number} count
 * Output: {{ current: HTMLButtonElement|null }[]}
 * Purpose: 셀 상세 보기 버튼 포커스 복귀용 ref 묶음을 생성한다.
 */
const createTriggerRefs = (count) => Array.from({ length: count }, () => createRef());

/**
 * Input: {{
 *   result: {
 *     yolo_crops: string[],
 *     denoised: string[],
 *     sr: string[]
 *   },
 *   previews: (string|null)[],
 *   onDetailOpen: (info:{stage:string, index:number, triggerRef:{ current: HTMLButtonElement|null }})=>void
 * }}
 * Output: {JSX.Element}
 * Purpose: 4행 5열 파이프라인 결과 그리드와 상세 버튼을 렌더링한다.
 */
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
        <p className={styles.text}>입력과 각 후처리 단계를 열별로 나란히 비교할 수 있습니다.</p>
      </div>

      <div className={styles.rows}>
        {ROWS.map((row) => {
          const images = row.getImages({ result, previews });

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
