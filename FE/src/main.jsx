import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

/**
 * Input: 없음.
 * Output: {void}
 * Purpose: React 앱 루트를 생성하고 App 컴포넌트를 렌더링한다.
 */
const renderApp = () => {
  ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
};

renderApp();
