import { useRef } from 'react';
import styles from './ImageSlot.module.css';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

/**
 * Input: {File|undefined|null} file
 * Output: {boolean}
 * Purpose: 개별 슬롯에 허용된 이미지 파일인지 검사한다.
 */
const isValidImageFile = (file) => Boolean(file && ACCEPTED_TYPES.includes(file.type));

/**
 * Input: {{
 *   index: number,
 *   file: File|null,
 *   preview: string|null,
 *   onSlotChange: (index:number, file:File)=>void,
 *   onSlotClear: (index:number)=>void
 * }}
 * Output: {JSX.Element}
 * Purpose: 단일 업로드 슬롯의 업로드, 교체, 삭제 UI를 제공한다.
 */
export default function ImageSlot({ index, file, preview, onSlotChange, onSlotClear }) {
  const inputRef = useRef(null);

  /**
   * Input: {File|undefined|null} nextFile
   * Output: {void}
   * Purpose: 유효한 파일만 상위 슬롯 변경 핸들러로 전달한다.
   */
  const commitFile = (nextFile) => {
    if (!isValidImageFile(nextFile)) {
      return;
    }

    onSlotChange(index, nextFile);
  };

  /**
   * Input: {React.ChangeEvent<HTMLInputElement>} event
   * Output: {void}
   * Purpose: 파일 선택 input 변경을 처리한다.
   */
  const handleChange = (event) => {
    commitFile(event.target.files?.[0]);
    event.target.value = '';
  };

  /**
   * Input: {React.DragEvent<HTMLDivElement>} event
   * Output: {void}
   * Purpose: 슬롯 드롭 타깃의 dragover 기본 동작을 차단한다.
   */
  const handleDragOver = (event) => {
    event.preventDefault();
  };

  /**
   * Input: {React.DragEvent<HTMLDivElement>} event
   * Output: {void}
   * Purpose: 슬롯에 드롭된 첫 이미지 파일로 현재 슬롯을 교체한다.
   */
  const handleDrop = (event) => {
    event.preventDefault();
    commitFile(event.dataTransfer.files?.[0]);
  };

  return (
    <div
      className={styles.slot}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        className={styles.hiddenInput}
        type="file"
        accept=".jpg,.jpeg,.png,.webp"
        onChange={handleChange}
      />

      {preview ? (
        <>
          <img className={styles.preview} src={preview} alt={`input-slot-${index + 1}`} />
          <div className={styles.overlay}>
            <span className={styles.overlayText}>교체</span>
          </div>
        </>
      ) : (
        <div className={styles.placeholder}>
          <span className={styles.slotLabel}>Slot {index + 1}</span>
          <span className={styles.slotHint}>이미지를 올리거나 클릭하세요</span>
        </div>
      )}

      {file ? (
        <button
          type="button"
          className={styles.clearButton}
          aria-label={`clear-slot-${index + 1}`}
          onClick={(event) => {
            event.stopPropagation();
            onSlotClear(index);
          }}
        >
          ×
        </button>
      ) : null}
    </div>
  );
}
