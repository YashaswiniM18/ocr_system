\# Real-Time Document OCR \& Image Asset Extraction



A FastAPI-based backend system that performs \*\*OCR, document classification, field extraction, and image asset extraction (Face \& Signature)\*\* for Indian documents like:



\- Aadhaar Card

\- PAN Card

\- Driving License (DL)

\- Marksheet



The system automatically detects the document type, extracts structured metadata, and correctly crops face and signature images.



---



\## Features



\- 📄 Document Classification (Aadhaar / PAN / DL / Marksheet)

\- 🔍 OCR using PaddleOCR

\- 🧠 Intelligent Field Extraction

\- 🖼️ Face Image Extraction

\- ✍️ Signature Extraction (PAN \& DL only)

\- 🔄 Handles rotated images

\- ⚡ FastAPI backend with REST API

\- 🌐 CORS enabled for frontend integration



---



\## Supported Extractions



\### Aadhaar

\- Name

\- Date of Birth 

\- Gender

\- Aadhaar Number

\- Face Image



\### PAN

\- Name

\- Father’s Name

\- PAN Number

\- Date of Birth

\- Face Image

\- Signature Image



\### Driving License

\- Name

\- License Number

\- Date of Birth

\- Face Image

\- Signature Image



\### Marksheet

\- Candidate Name

\- Candidate ID / Roll Number

\- Board Name

\- Subject-wise Marks

\- Total Marks

\- Result (PASS / FAIL)

\- Face Image



---



\## Tech Stack



\- \*\*Backend:\*\* FastAPI

\- \*\*OCR Engine:\*\* PaddleOCR

\- \*\*Image Processing:\*\* OpenCV, Pillow

\- \*\*Language:\*\* Python 3

\- \*\*Frontend:\*\* React (served separately)



---







