import { useState, useRef } from 'react';
import styles from './VideoTrimmer.module.css';

export default function VideoTrimmer({ videoFile, videoPreview, onTimeRangeChange, onClearVideo }) {
  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);

  const handleLoadedMetadata = () => {
    const vidDuration = videoRef.current.duration;
    setDuration(vidDuration);
    
    // Default 3 second range from the beginning
    const defaultEnd = Math.min(3, vidDuration);
    setStartTime(0);
    setEndTime(defaultEnd);
    onTimeRangeChange(0, defaultEnd);
  };

  const handleSetStart = () => {
    if (!videoRef.current) return;
    const curr = videoRef.current.currentTime;
    setStartTime(curr);
    
    // Ensure end time is always after start time, auto-adjusting to 3 seconds if needed
    let newEnd = endTime;
    if (curr >= endTime) {
      newEnd = Math.min(curr + 3, duration);
      setEndTime(newEnd);
    }
    onTimeRangeChange(curr, newEnd);
  };

  const handleSetEnd = () => {
    if (!videoRef.current) return;
    const curr = videoRef.current.currentTime;
    setEndTime(curr);
    
    // Ensure start time is always before end time
    let newStart = startTime;
    if (curr <= startTime) {
      newStart = Math.max(curr - 3, 0);
      setStartTime(newStart);
    }
    onTimeRangeChange(newStart, curr);
  };

  const handleStartChange = (e) => {
    const val = Number(e.target.value);
    setStartTime(val);
    onTimeRangeChange(val, endTime);
  };

  const handleEndChange = (e) => {
    const val = Number(e.target.value);
    setEndTime(val);
    onTimeRangeChange(startTime, val);
  };

  return (
    <div className={styles.trimmerWrapper}>
      <div className={styles.header}>
        <span>🎥 {videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(2)} MB)</span>
        <button onClick={onClearVideo} className={styles.clearBtn}>영상 지우기</button>
      </div>
      
      <video 
        ref={videoRef} 
        src={videoPreview} 
        controls 
        onLoadedMetadata={handleLoadedMetadata}
        className={styles.videoPlayer}
      />
      
      <div className={styles.controls}>
        <p style={{ margin: '0 0 8px 0', fontWeight: 'bold', color: '#fff' }}>✂️ 분석할 구간을 선택하세요 (3초 이내 권장)</p>
        <p style={{ margin: 0, fontSize: '13px', color: '#aaa' }}>영상을 재생하다가 원하는 장면에서 [현재 화면을 ...] 버튼을 누르세요.</p>
        
        <div className={styles.timeInputs}>
          <div className={styles.timeBox}>
            <label>시작 시간 (초)</label>
            <input type="number" step="0.1" min="0" max={duration} value={startTime.toFixed(1)} onChange={handleStartChange} />
            <button onClick={handleSetStart}>현재 화면을 시작으로</button>
          </div>
          <div className={styles.timeBox}>
            <label>종료 시간 (초)</label>
            <input type="number" step="0.1" min="0" max={duration} value={endTime.toFixed(1)} onChange={handleEndChange} />
            <button onClick={handleSetEnd}>현재 화면을 종료로</button>
          </div>
        </div>
        
        <p className={styles.summary}>
          선택된 분석 구간: <strong style={{ color: '#4a90e2', fontSize: '20px' }}>{(endTime - startTime).toFixed(1)}초</strong>
        </p>
      </div>
    </div>
  );
}
