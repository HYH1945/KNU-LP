import styles from './OcrResult.module.css';

export default function OcrResult({ text, predictions = [] }) {
  return (
    <section className={styles.panel}>
      <p className={styles.label}>OCR RESULT</p>
      <div className={styles.value}>{text}</div>
      {predictions.length > 0 ? (
        <div className={styles.predictions}>
          {predictions.map((prediction, index) => (
            <span key={`${prediction}-${index}`} className={styles.prediction}>
              {prediction}
            </span>
          ))}
        </div>
      ) : null}
    </section>
  );
}
