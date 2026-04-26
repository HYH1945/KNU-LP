import { useRef } from 'react';
import ImageSlot from '../ImageSlot/ImageSlot';
import styles from './UploadZone.module.css';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

/**
 * Input: {File[]} fileList
 * Output: {File[]}
 * Purpose: 허용된 MIME 타입의 이미지 파일만 필터링한다.
 */
const getValidFiles = (fileList) =>
  fileList.filter((file) => ACCEPTED_TYPES.includes(file.type));

/**
 * Input: {(File|null)[]} files, {File[]} incomingFiles, {(index:number, file:File)=>void} onSlotChange
 * Output: {void}
 * Purpose: 빈 슬롯 순서대로만 파일을 배치 업로드한다.
 */
const fillEmptySlots = (files, incomingFiles, onSlotChange) => {
  let incomingIndex = 0;

  files.forEach((file, slotIndex) => {
    if (file !== null || incomingIndex >= incomingFiles.length) {
      return;
    }

    onSlotChange(slotIndex, incomingFiles[incomingIndex]);
    incomingIndex += 1;
  });
};

/**
 * Input: {{
 *   files: (File|null)[],
 *   previews: (string|null)[],
 *   onSlotChange: (index:number, file:File)=>void,
 *   onSlotClear: (index:number)=>void
 * }}
 * Output: {JSX.Element}
 * Purpose: 배치 업로드 영역과 5개 개별 슬롯을 함께 렌더링한다.
 */
export default function UploadZone({ files, previews, onSlotChange, onSlotClear }) {
  const batchInputRef = useRef(null);
  const filledCount = files.filter((file) => file !== null).length;

  /**
   * Input: {File[]} incomingFiles
   * Output: {void}
   * Purpose: 유효한 이미지 파일을 추려 빈 슬롯에만 반영한다.
   */
  const handleIncomingFiles = (incomingFiles) => {
    if (filledCount === files.length) {
      return;
    }

    const validFiles = getValidFiles(incomingFiles);
    fillEmptySlots(files, validFiles, onSlotChange);
  };

  /**
   * Input: {React.ChangeEvent<HTMLInputElement>} event
   * Output: {void}
   * Purpose: 배치 파일 선택 input의 변경 이벤트를 처리한다.
   */
  const handleBatchChange = (event) => {
    handleIncomingFiles(Array.from(event.target.files ?? []));
    event.target.value = '';
  };

  /**
   * Input: {React.DragEvent<HTMLDivElement>} event
   * Output: {void}
   * Purpose: 배치 드롭 영역의 dragover 기본 동작을 차단한다.
   */
  const handleDragOver = (event) => {
    event.preventDefault();
  };

  /**
   * Input: {React.DragEvent<HTMLDivElement>} event
   * Output: {void}
   * Purpose: 드롭된 이미지 파일들을 빈 슬롯에 순서대로 반영한다.
   */
  const handleDrop = (event) => {
    event.preventDefault();
    handleIncomingFiles(Array.from(event.dataTransfer.files ?? []));
  };

  return (
    <div className={styles.wrapper}>
      <div
        className={styles.dropZone}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        onClick={() => batchInputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            batchInputRef.current?.click();
          }
        }}
      >
        <input
          ref={batchInputRef}
          className={styles.hiddenInput}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.webp"
          onChange={handleBatchChange}
        />
        <p className={styles.dropTitle}>이미지 여러 장을 한 번에 업로드</p>
        <p className={styles.dropText}>
          드래그 앤 드롭 또는 클릭으로 선택하세요. 비어 있는 슬롯만 순서대로 채웁니다.
        </p>
      </div>

      <div className={styles.slotGrid}>
        {files.map((file, index) => (
          <ImageSlot
            key={index}
            index={index}
            file={file}
            preview={previews[index]}
            onSlotChange={onSlotChange}
            onSlotClear={onSlotClear}
          />
        ))}
      </div>

      {filledCount < files.length ? (
        <p className={styles.helper}>5장을 업로드해야 합니다</p>
      ) : null}
    </div>
  );
}
