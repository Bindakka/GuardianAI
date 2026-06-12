# 🛡️ GuardianAI — AI Fraud Detection Platform

> **Google Cloud AI Hackathon** | Track: AI for Social Good | Partner: MongoDB

## Live Demo
`https://your-cloudrun-url.run.app` ← replace after deploying

## Problem
- ₹11,333 Crore lost to cyber fraud in India (2023, RBI)
- 7.4M+ cybercrime complaints in India in 2023
- Citizens have no fast, accessible fraud detection tool

## Solution
5-agent AI platform: Face Verification, Fraud Detection, Risk Scoring, PDF Report Generator, Knowledge Agent — all powered by Google Gemini, stored in MongoDB Atlas, hosted on Cloud Run.

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Backend | Python Flask |
| AI | **Google Gemini 1.5 Flash** (FREE) |
| Database | **MongoDB Atlas** (FREE) |
| Deploy | **Google Cloud Run** |
| PDF | ReportLab |

## Local Setup

### 1. Get free API keys
- Gemini: https://makersuite.google.com/app/apikey
- MongoDB: https://www.mongodb.com/atlas (free cluster)

### 2. Install and run
```bash
pip install -r requirements.txt

# Windows
set GEMINI_API_KEY=your_key_here
set MONGODB_URI=your_mongodb_uri_here

# Mac/Linux
export GEMINI_API_KEY=your_key_here
export MONGODB_URI=your_mongodb_uri_here

python app.py
# Open http://localhost:8080
```

## Cloud Run Deploy
```bash
docker build -t guardianai .
docker tag guardianai gcr.io/YOUR_PROJECT_ID/guardianai
docker push gcr.io/YOUR_PROJECT_ID/guardianai
gcloud run deploy guardianai \
  --image gcr.io/YOUR_PROJECT_ID/guardianai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=YOUR_KEY,MONGODB_URI=YOUR_URI
```

## API Endpoints
| Method | Endpoint | Agent |
|--------|----------|-------|
| POST | `/api/face-verify` | Face Verification Agent |
| POST | `/api/fraud-detection` | Fraud Detection Agent |
| POST | `/api/risk-score` | Risk Scoring Agent |
| POST | `/api/generate-report` | Report Generation Agent (PDF) |
| POST | `/api/knowledge/search` | Knowledge Agent |
| GET  | `/api/knowledge/stats` | Knowledge Agent stats |





## License
MIT
