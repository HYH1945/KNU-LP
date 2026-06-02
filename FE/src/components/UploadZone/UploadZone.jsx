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
  onTimeRangeChange,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const selected = Array.from(event.target.files ?? []);
    if (selected.length === 0) return;

    if (inputMode === 'image') {
      const validImages = selected.filter((file) => ACCEPTED_IMAGES.includes(file.type));
      if (validImages.length > 0) {
        onAddImages(validImages);
      }
    } else {
      const validVideo = selected.find((file) => ACCEPTED_VIDEOS.includes(file.type));
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
      const validImages = dropped.filter((file) => ACCEPTED_IMAGES.includes(file.type));
      if (validImages.length > 0) onAddImages(validImages);
    } else {
      const validVideo = dropped.find((file) => ACCEPTED_VIDEOS.includes(file.type));
      if (validVideo) onSetVideo(validVideo);
    }
  };

  return (
    <div className={styles.wrapper}>
      {inputMode === 'image' && files.length > 0 && (
        <div className={styles.previewGrid}>
          {files.map((file, index) => (
            <div key={`${file.name}-${index}`} className={styles.previewItem}>
              <img src={previews[index]} alt={`preview-${index + 1}`} className={styles.previewImage} />
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onRemoveImage(index);
                }}
                className={styles.removeButton}
                aria-label={`remove-image-${index + 1}`}
              >
                x
              </button>
            </div>
          ))}
        </div>
      )}

      {inputMode === 'video' && videoFile && (
        <VideoTrimmer
          videoFile={videoFile}
          videoPreview={videoPreview}
          onTimeRangeChange={onTimeRangeChange}
          onClearVideo={onClearVideo}
        />
      )}

      {!(inputMode === 'video' && videoFile) && (
        <div
          className={styles.dropZone}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              fileInputRef.current?.click();
            }
          }}
        >
          <input
            ref={fileInputRef}
            className={styles.hiddenInput}
            type="file"
            multiple={inputMode === 'image'}
            accept={inputMode === 'image' ? ACCEPTED_IMAGES.join(',') : ACCEPTED_VIDEOS.join(',')}
            onChange={handleFileChange}
          />
          <p className={styles.dropTitle}>
            {inputMode === 'image' ? '이미지 파일 추가' : '동영상 파일 업로드'}
          </p>
          <p className={styles.dropText}>
            {inputMode === 'image'
              ? 'JPEG, PNG, WEBP 이미지를 여러 장 선택할 수 있습니다.'
              : 'MP4, WEBM, OGG 영상을 하나 선택할 수 있습니다.'}
          </p>
        </div>
      )}
    </div>
  );
}
