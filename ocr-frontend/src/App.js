import React, { useState } from 'react';
import './App.css';
import logo from './assets/logo.png';

const API_BASE = "http://localhost:8000";

// Updated steps to match your Internship Project flow
const STEPS = ["UPLOAD", "PREPROCESS", "ANALYSIS", "EXTRACTION"];

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const startOCR = async () => {
    if (!file) return;

    setLoading(true);
    setCurrentStep(0); // Step 1: Uploading
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 300s timeout for first-time model load

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        let msg = "Backend error";
        try {
          const errBody = await response.json();
          msg = errBody.detail || msg;
        } catch (e) { }
        throw new Error(msg);
      }
      const data = await response.json();

      // Step 2: Automated Cleaning (clarity, rotation, noise)
      setCurrentStep(1);
      await new Promise(r => setTimeout(r, 1200));

      // Step 3: AI Text Extraction (Name, DOB, Marks)
      setCurrentStep(2);
      await new Promise(r => setTimeout(r, 1200));

      // Step 4: Final Extraction (Face & Signature cropping)
      setCurrentStep(3);
      await new Promise(r => setTimeout(r, 800));

      setResult(data);
    } catch (err) {
      console.error(err);
      setError(`Failed to process document: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <nav className="nav-header">
        <div className="logo">
          <img src={logo} alt="EtherX Logo" className="nav-logo-img" />
          EtherXVision
        </div>
        <div className="secure-tag">SECURE V1.0</div>
      </nav>

      <div className="container">
        <div className="glass-panel">
          <header className="panel-header">
            <h1 className="main-title">Image <span>Scanner</span></h1>
            <p className="sub-text">Real-Time OCR & Extraction Engine</p>
          </header>

          {/* 1. INITIAL UPLOAD UI */}
          {!loading && !result && (
            <div className="upload-section">
              <div className="upload-zone">
                <div className="upload-content">
                  <div className="icon-upload">📁</div>
                  <p className="upload-label">
                    {file ? `Selected: ${file.name}` : "Drop Identity Document (JPG/PNG)"}
                  </p>
                  <div className="btn-choose">Choose File</div>
                </div>
                <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} className="file-input" />
              </div>
              <button className="btn-scan" onClick={startOCR} disabled={!file}>INITIATE SCAN</button>
            </div>
          )}

          {/* 2. PROGRESS TRACKER (Active after clicking Initiate Scan) */}
          {loading && (
            <div className="progress-container">
              <div className="generating-box">
                <div className="gen-text">
                  <h3>Generating Results</h3>
                  <p className="sub-text-small">
                    {currentStep === 1 && "Fixing rotation, reducing noise, and improving clarity..."}
                    {currentStep === 2 && "Reading text and extracting key document details..."}
                    {currentStep === 3 && "Auto-cropping face photo and signature assets..."}
                  </p>
                </div>
              </div>

              <div className="stepper">
                {STEPS.map((step, index) => (
                  <div key={step} className={`step-item ${index <= currentStep ? 'active' : ''}`}>
                    <div className="step-dot"></div>
                    <span className="step-label">{step}</span>
                    {index < STEPS.length - 1 && <div className="step-line"></div>}
                  </div>
                ))}
              </div>
              <div className="spinner-loader"></div>
            </div>
          )}

          {/* 3. FINAL RESULT UI */}
          {result && (
            <>
              <div
                className="result-display"
                style={(!result.face_image && !result.signature_image) ? { gridTemplateColumns: '1fr', maxWidth: '700px', margin: '50px auto 0' } : {}}
              >
                <div className="metadata-box">
                  <h3 className="section-label">EXTRACTED DATA</h3>
                  <pre className="data-preview">{JSON.stringify(result.extracted_fields, null, 2)}</pre>
                </div>

                {(result.face_image || result.signature_image) && (
                  <div className="assets-box">
                    <h3 className="section-label">EXTRACTED ASSETS</h3>
                    <div className="asset-row">
                      {result.face_image && (
                        <div className="asset-card">
                          <span className="asset-label">PHOTO</span>
                          <img src={`${API_BASE}/${result.face_image}`} alt="Face" />
                        </div>
                      )}
                      {result.signature_image && (
                        <div className="asset-card">
                          <span className="asset-label">SIGNATURE</span>
                          <img src={`${API_BASE}/${result.signature_image}`} alt="Signature" />
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              <div style={{ textAlign: 'center', marginTop: '40px' }}>
                <button className="btn-scan" onClick={() => { setResult(null); setFile(null); }}>NEW SCAN</button>
              </div>
            </>
          )}

          {error && <p className="error-text">{error}</p>}
        </div>

        <footer className="external-footer">
          <p className="powered-by">© 2026 EtherX Innovations. All rights reserved.
          </p>
          <img src={logo} alt="EtherX Logo" className="footer-logo-visible" />
        </footer>
      </div>
    </div>
  );
}

export default App;