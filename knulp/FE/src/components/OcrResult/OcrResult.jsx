import styles from './OcrResult.module.css';

export default function OcrResult({ text }) {
  return (
    <section className={styles.panel}>
      <p className={styles.label}>OCR RESULT</p>
      <div className={styles.value}>{text}</div>
    </section>
  );
}
