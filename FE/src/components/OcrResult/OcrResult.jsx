import styles from './OcrResult.module.css';

/**
 * Input: {{ text: string }}
 * Output: {JSX.Element}
 * Purpose: OCR 문자열 결과를 단독 배너 형태로 표시한다.
 */
export default function OcrResult({ text }) {
  return (
    <section className={styles.panel}>
      <p className={styles.label}>OCR RESULT</p>
      <div className={styles.value}>{text}</div>
    </section>
  );
}
