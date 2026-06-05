import { useState } from 'react';
import styles from './TargetSelector.module.css';

export default function TargetSelector({ frames, onSelectionsComplete, onCancel }) {
  // selections: { frameIndex: bboxIndex }
  const [selections, setSelections] = useState({});

  const handleBboxClick = (frameIndex, bboxIndex) => {
    setSelections(prev => ({ ...prev, [frameIndex]: bboxIndex }));
  };

  const isComplete = frames && frames.length > 0 && Object.keys(selections).length === frames.length;

  const handleSubmit = () => {
    if (isComplete) {
      // 선택된 바운딩 박스들만 모아서 상위 컴포넌트로 전달
      const selectedBboxes = frames.map((frame, index) => frame.bboxes[selections[index]]);
      onSelectionsComplete(selectedBboxes);
    }
  };

  if (!frames || frames.length === 0) return null;

  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <h2>🎯 추적할 번호판 선택 (총 {frames.length}장)</h2>
        <p>각 프레임에서 복원하고 싶은 <strong>동일한 차량의 번호판</strong>을 하나씩 클릭해주세요.</p>
      </div>

      <div className={styles.framesGrid}>
        {frames.map((frame, fIndex) => (
          <div key={frame.id || fIndex} className={styles.frameContainer}>
            <p className={styles.frameLabel}>
              Frame {fIndex + 1} {selections[fIndex] !== undefined ? '✅' : '❌'}
            </p>
            <div className={styles.imageWrapper}>
              <img src={frame.image_url} alt={`Frame ${fIndex + 1}`} className={styles.image} />
              
              {frame.bboxes && frame.bboxes.map((bbox, bIndex) => {
                const isSelected = selections[fIndex] === bIndex;
                return (
                  <div
                    key={bbox.id || bIndex}
                    className={`${styles.bbox} ${isSelected ? styles.selected : ''}`}
                    style={{
                      left: `${bbox.x}%`,
                      top: `${bbox.y}%`,
                      width: `${bbox.w}%`,
                      height: `${bbox.h}%`
                    }}
                    onClick={() => handleBboxClick(fIndex, bIndex)}
                  >
                    {isSelected && <span className={styles.checkIcon}>✔️</span>}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className={styles.footer}>
        <button className={styles.cancelBtn} onClick={onCancel}>돌아가기</button>
        <button 
          className={styles.submitBtn} 
          disabled={!isComplete}
          onClick={handleSubmit}
        >
          선택 완료 및 최종 분석 시작 🚀
        </button>
      </div>
    </div>
  );
}
