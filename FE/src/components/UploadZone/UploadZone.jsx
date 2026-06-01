import { useRef } from 'react';
import VideoTrimmer from '../VideoTrimmer/VideoTrimmer';
import styles from './UploadZone.module.css';

const ACCEPTED_IMAGES = ['image/jpeg', 'image/png', 'image/webp'];
const ACCEPTED_VIDEOS = ['video/mp4', 'video/webm', 'video/ogg'];

export default function UploadZone({
  inputMode,
  files,
  previews,
  videoFile,
  videoPreview,
  onAddImages,
  onRemoveImage,
  onSetVideo,
  onClearVideo,
  onTimeRangeChange
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const selected = Array.from(event.target.files ?? []);
    if (selected.length === 0) return;

    if (inputMode === 'image') {
      const validImages = selected.filter(f => ACCEPTED_IMAGES.includes(f.type));
      if (validImages.length > 0) {
        onAddImages(validImages);
      }
    } else {
      const validVideo = selected.find(f => ACCEPTED_VIDEOS.includes(f.type));
      if (validVideo) {
        onSetVideo(validVideo);
      }
    }
    event.target.value = '';
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const dropped = Array.from(event.dataTransfer.files ?? []);
    if (dropped.length === 0) return;

    if (inputMode === 'image') {
      const validImages = dropped.filter(f => ACCEPTED_IMAGES.includes(f.type));
      if (validImages.length > 0) onAddImages(validImages);
    } else {
      const validVideo = dropped.find(f => ACCEPTED_VIDEOS.includes(f.type));
      if (validVideo) onSetVideo(validVideo);
    }
  };

  return (
    <div className={styles.wrapper}>
      {/* 이미지 모드일 때: 기존 이미지들을 썸네일로 나열 */}
      {inputMode === 'image' && files.length > 0 && (
        <div style={{ marginBottom: '16px', display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
          {files.map((file, index) => (
            <div key={index} style={{ position: 'relative', width: '120px', height: '120px', border: '1px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
              <img src={previews[index]} alt="preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              <button 
                onClick={(e) => { e.stopPropagation(); onRemoveImage(index); }}
                style={{ position: 'absolute', top: 4, right: 4, background: 'rgba(255,0,0,0.8)', color: 'white', border: 'none', cursor: 'pointer', borderRadius: '4px', padding: '2px 6px', fontSize: '12px', fontWeight: 'bold' }}
              >
                X
              </button>
            </div>
          ))}
        </div>
      )}

      {/* 동영상 모드일 때: 등록된 동영상 표시 및 구간 자르기 UI */}
      {inputMode === 'video' && videoFile && (
        <VideoTrimmer 
          videoFile={videoFile} 
          videoPreview={videoPreview} 
          onTimeRangeChange={onTimeRangeChange} 
          onClearVideo={onClearVideo} 
        />
      )}

      {/* 추가 업로드 (드롭 존) 영역 */}
      {!(inputMode === 'video' && videoFile) && (
        <div
          className={styles.dropZone}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          style={{ cursor: 'pointer', border: '2px dashed #666', padding: '32px', textAlign: 'center', borderRadius: '8px', background: 'transparent' }}
        >
          <input
            ref={fileInputRef}
            className={styles.hiddenInput}
            type="file"
            multiple={inputMode === 'image'}
            accept={inputMode === 'image' ? ACCEPTED_IMAGES.join(',') : ACCEPTED_VIDEOS.join(',')}
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <p className={styles.dropTitle} style={{ fontSize: '18px', margin: '0 0 8px 0', color: '#fff' }}>
            {inputMode === 'image' ? '🖼️ 이미지 파일 추가 (여러 장 가능)' : '🎥 동영상 파일(.mp4) 업로드'}
          </p>
          <p className={styles.dropText} style={{ margin: 0, color: '#aaa', fontSize: '14px' }}>
            여기로 파일을 드래그하거나 클릭하여 선택하세요.
          </p>
        </div>
      )}
    </div>
  );
}
