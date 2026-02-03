import React, { useState } from 'react';
import './App.css';
import logo from './assets/logo.jpeg';

const API_BASE = "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const startOCR = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/upload/`, { 
        method: 'POST', 
        body: formData 
      });
      const data = await response.json();
      setResult(data);
    } catch (e) {
      console.error("OCR System Error:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      {/* Top Navigation */}
      <nav className="nav-header">
        <div className="logo">OCR SYSTEM</div>
        <div className="secure-tag">SECURE V1.0</div>
      </nav>

      <div className="container">
        {/* Main Scanner Panel */}
        <div className="glass-panel">
          <header className="panel-header">
            <h1 className="main-title">Image <span>Scanner</span></h1>
            <p className="sub-text">Real-time OCR & Image Extraction Engine</p>
          </header>

          <div className="upload-zone">
            <div className="upload-content">
              <div className="icon-upload">📁</div>
              <p className="upload-label">
                {file ? `Selected: ${file.name}` : "Drop Identity Document (JPG / PNG)"}
              </p>
              {/* Manual Choose File Button UI */}
              <div className="btn-choose">Choose File</div>
            </div>
            <input 
              type="file" 
              onChange={(e) => setFile(e.target.files[0])}
              className="file-input"
            />
          </div>

          <button 
            className="btn-scan" 
            onClick={startOCR} 
            disabled={loading || !file}
          >
            {loading ? "SCANNING ASSETS..." : "INITIATE SCAN"}
          </button>

          {/* Results Display Area */}
          {result && (
            <div className="result-display">
              <div className="metadata-box">
                <h3 className="section-label">EXTRACTED METADATA</h3>
                <pre className="data-preview">
                  {JSON.stringify(result.fields, null, 2)}
                </pre>
              </div>
              
              <div className="assets-box">
                <h3 className="section-label">IMAGE ASSETS</h3>
                <div className="asset-row">
                  <div className="asset-card">
                    <span>FACE</span>
                    <img src={`${API_BASE}/${result.face_image}`} alt="Face" />
                  </div>
                  <div className="asset-card">
                    <span>SIGNATURE</span>
                    <img src={`${API_BASE}/${result.signature_image}`} alt="Sign" />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Enhanced Aesthetic Footer */}
        <footer className="external-footer">
          <p className="powered-by">
            Powered by <span>ETHERX INNOVATIONS</span>
          </p>
          <img src={logo} alt="EtherX Logo" className="footer-logo-visible" />
        </footer>
      </div>
    </div>
  );
}

export default App;