import { useEffect, useRef, useState } from 'react';
import UploadZone from './components/UploadZone/UploadZone';
import TargetSelector from './components/TargetSelector/TargetSelector';
import ResultGrid from './components/ResultGrid/ResultGrid';
import OcrResult from './components/OcrResult/OcrResult';
import DetailModal from './components/DetailModal/DetailModal';
import { analyzeImages } from './services/api';
import styles from './App.module.css';

const DUMMY_FRAMES = [
  {
    id: 'f1', image_url: 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=640&q=80',
    bboxes: [ { id: 'b1', x: 20, y: 60, w: 15, h: 10 }, { id: 'b2', x: 60, y: 70, w: 10, h: 8 } ]
  },
  {
    id: 'f2', image_url: 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=640&q=80',
    bboxes: [ { id: 'b1', x: 25, y: 62, w: 15, h: 10 }, { id: 'b2', x: 65, y: 68, w: 10, h: 8 } ]
  },
  {
    id: 'f3', image_url: 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=640&q=80',
    bboxes: [ { id: 'b1', x: 30, y: 65, w: 15, h: 10 }, { id: 'b2', x: 70, y: 65, w: 10, h: 8 } ]
  },
  {
    id: 'f4', image_url: 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=640&q=80',
    bboxes: [ { id: 'b1', x: 35, y: 68, w: 15, h: 10 }, { id: 'b2', x: 75, y: 60, w: 10, h: 8 } ]
  },
  {
    id: 'f5', image_url: 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=640&q=80',
    bboxes: [ { id: 'b1', x: 40, y: 70, w: 15, h: 10 }, { id: 'b2', x: 80, y: 55, w: 10, h: 8 } ]
  }
];

export default function App() {
  const [appStep, setAppStep] = useState('upload'); // 'upload' | 'target_selection' | 'result'
  const [inputMode, setInputMode] = useState('image'); // 'image' | 'video'
  
  // Image Mode State
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  
  // Video Mode State
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [videoTimeRange, setVideoTimeRange] = useState({ start: 0, end: 3 });

  // 2-Step Communication State
  const [candidateFrames, setCandidateFrames] = useState([]);

  // Result State
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modalInfo, setModalInfo] = useState(null);

  // Cleanup ObjectURLs to prevent memory leaks
  useEffect(() => {
    return () => {
      previews.forEach(p => URL.revokeObjectURL(p));
      if (videoPreview) URL.revokeObjectURL(videoPreview);
    };
  }, [previews, videoPreview]);

  const addImages = (newFiles) => {
    const newPreviews = newFiles.map(f => URL.createObjectURL(f));
    setFiles(prev => [...prev, ...newFiles]);
    setPreviews(prev => [...prev, ...newPreviews]);
    resetResults();
  };

  const removeImage = (index) => {
    URL.revokeObjectURL(previews[index]);
    setFiles(prev => prev.filter((_, i) => i !== index));
    setPreviews(prev => prev.filter((_, i) => i !== index));
    resetResults();
  };

  const setVideo = (file) => {
    if (videoPreview) URL.revokeObjectURL(videoPreview);
    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file));
    resetResults();
  };
  
  const clearVideo = () => {
    if (videoPreview) URL.revokeObjectURL(videoPreview);
    setVideoFile(null);
    setVideoPreview(null);
    resetResults();
  };

  const resetResults = () => {
    setResult(null);
    setError(null);
    setStatus('idle');
    setAppStep('upload');
  };

  // 1단계: 분석 시작 (동영상 구간 전송 및 프레임 추출)
  const handleAnalyzeUpload = () => {
    setStatus('loading');
    setError(null);
    
    setTimeout(() => {
      setStatus('idle');
      if (inputMode === 'video') {
        setCandidateFrames(DUMMY_FRAMES); // 백엔드 통신 모방
        setAppStep('target_selection');
      } else {
        // 이미지 모드는 바로 결과 처리 (추후 구현)
        alert('이미지 분석 시작!');
      }
    }, 1000);
  };

  // 2단계: 유저가 5프레임 타겟 지정을 완료하고 최종 전송
  const handleTargetSelectionComplete = (selectedBboxes) => {
    setAppStep('upload');
    setStatus('loading');
    console.log("선택된 바운딩 박스들:", selectedBboxes);
    
    // (이후 Step 1.4 에서 백엔드로 이거 전송하고 ResultGrid로 응답 넘길 예정)
    setTimeout(() => {
      setStatus('done');
      alert('최종 분석이 백그라운드에서 진행중입니다! (API 구현 대기 중)');
    }, 1500);
  };

  const canAnalyze = inputMode === 'image' ? files.length > 0 : videoFile !== null;

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <header className={styles.header}>
          <p className={styles.eyebrow}>AI LICENSE PLATE DEMO</p>
          <h1 className={styles.title}>Multi-Frame Target Inspector</h1>
          <p className={styles.description}>
            여러 장의 이미지 또는 동영상을 업로드하여 다중 프레임 기반의 번호판 복원(MFSR) 파이프라인을 시연합니다.
          </p>
        </header>

        <main className={styles.content}>
          {appStep === 'upload' && (
            <section className={styles.panel}>
              <div className={styles.panelHeader}>
                <div>
                  <h2 className={styles.panelTitle}>Upload Media</h2>
                  <div style={{ marginTop: '8px', display: 'flex', gap: '12px', fontSize: '14px', fontWeight: 'bold' }}>
                    <label style={{ cursor: 'pointer' }}>
                      <input type="radio" name="mode" checked={inputMode === 'image'} onChange={() => { setInputMode('image'); resetResults(); }} /> 이미지 모드 (N장)
                    </label>
                    <label style={{ cursor: 'pointer' }}>
                      <input type="radio" name="mode" checked={inputMode === 'video'} onChange={() => { setInputMode('video'); resetResults(); }} /> 동영상 모드 (1개)
                    </label>
                  </div>
                </div>
                <button
                  type="button"
                  className={styles.analyzeButton}
                  disabled={!canAnalyze}
                  onClick={handleAnalyzeUpload}
                >
                  {status === 'loading' ? '처리 중...' : '프레임 추출 시작'}
                </button>
              </div>

              <UploadZone
                inputMode={inputMode}
                files={files}
                previews={previews}
                videoFile={videoFile}
                videoPreview={videoPreview}
                onAddImages={addImages}
                onRemoveImage={removeImage}
                onSetVideo={setVideo}
                onClearVideo={clearVideo}
                onTimeRangeChange={(start, end) => setVideoTimeRange({ start, end })}
              />

              {status === 'error' ? <p className={styles.errorBanner}>{error}</p> : null}
            </section>
          )}

          {appStep === 'target_selection' && (
            <TargetSelector 
              frames={candidateFrames} 
              onSelectionsComplete={handleTargetSelectionComplete} 
              onCancel={() => { setAppStep('upload'); setStatus('idle'); }} 
            />
          )}

        </main>
      </div>
    </div>
  );
}
