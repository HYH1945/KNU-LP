import { useRef, useState } from 'react';
import styles from './VideoTrimmer.module.css';

export default function VideoTrimmer({ videoFile, videoPreview, onTimeRangeChange, onClearVideo }) {
  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);

  const handleLoadedMetadata = () => {
    const videoDuration = videoRef.current.duration;
    const defaultEnd = Math.min(3, videoDuration);
    setDuration(videoDuration);
    setStartTime(0);
    setEndTime(defaultEnd);
    onTimeRangeChange(0, defaultEnd);
  };

  const handleSetStart = () => {
    if (!videoRef.current) return;
    const current = videoRef.current.currentTime;
    setStartTime(current);

    let nextEnd = endTime;
    if (current >= endTime) {
      nextEnd = Math.min(current + 3, duration);
      setEndTime(nextEnd);
    }
    onTimeRangeChange(current, nextEnd);
  };

  const handleSetEnd = () => {
    if (!videoRef.current) return;
    const current = videoRef.current.currentTime;
    setEndTime(current);

    let nextStart = startTime;
    if (current <= startTime) {
      nextStart = Math.max(current - 3, 0);
      setStartTime(nextStart);
    }
    onTimeRangeChange(nextStart, current);
  };

  const handleStartChange = (event) => {
    const value = Number(event.target.value);
    setStartTime(value);
    onTimeRangeChange(value, endTime);
  };

  const handleEndChange = (event) => {
    const value = Number(event.target.value);
    setEndTime(value);
    onTimeRangeChange(startTime, value);
  };

  return (
    <div className={styles.trimmerWrapper}>
      <div className={styles.header}>
        <span>
          {videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(2)} MB)
        </span>
        <button type="button" onClick={onClearVideo} className={styles.clearBtn}>
          영상 지우기
        </button>
      </div>

      <video
        ref={videoRef}
        src={videoPreview}
        controls
        onLoadedMetadata={handleLoadedMetadata}
        className={styles.videoPlayer}
      />

      <div className={styles.controls}>
        <p className={styles.controlTitle}>분석 구간</p>

        <div className={styles.timeInputs}>
          <div className={styles.timeBox}>
            <label>시작 시간 (초)</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max={duration}
              value={startTime.toFixed(1)}
              onChange={handleStartChange}
            />
            <button type="button" onClick={handleSetStart}>
              현재 화면을 시작으로
            </button>
          </div>
          <div className={styles.timeBox}>
            <label>종료 시간 (초)</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max={duration}
              value={endTime.toFixed(1)}
              onChange={handleEndChange}
            />
            <button type="button" onClick={handleSetEnd}>
              현재 화면을 종료로
            </button>
          </div>
        </div>

        <p className={styles.summary}>선택 구간: {(endTime - startTime).toFixed(1)}초</p>
      </div>
    </div>
  );
}
