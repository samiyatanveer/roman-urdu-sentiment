
# Roman Urdu Sentiment Classifier API

A FastAPI-based REST API for sentiment classification in Roman Urdu (Urdu written in Latin characters, mixed with English). Uses a pretrained transformer model (`cardiffnlp/twitter-roberta-base-sentiment`) fine-tuned on multilingual social media text.

## Overview

**What this does:** Takes Roman Urdu/Hinglish text as input, classifies it as positive, negative, or neutral, and returns a confidence score.

**Example:**
```
Input:  "Yeh bilkul amazing hai"
Output: {"sentiment": "LABEL_2", "confidence": 0.94}
```

## Architecture

- **Model:** RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Training data:** Twitter sentiment dataset (100M+ tweets, multilingual)
- **Deployment:** FastAPI + Uvicorn
- **Input:** Roman Urdu/Hinglish text strings
- **Output:** JSON with sentiment label + confidence score (0-1)

## Quick Start

### Local

```bash
pip install -r requirements.txt
python -m uvicorn app_sentiment:app --reload
```

Visit: `http://127.0.0.1:8000/docs`

### Test

```bash
python inference.py
```

Runs 3 example predictions locally.

## API Endpoints

**POST `/predict`**
- Request: `{"text": "kya haal hai?"}`
- Response: `{"text": "...", "sentiment": "LABEL_1", "confidence": 0.69}`

**GET `/`**
- Returns: API status message

**GET `/docs`**
- Interactive Swagger UI for testing (FastAPI auto-generated)


## Files

- `app_sentiment.py` — FastAPI app with `/predict` endpoint
- `inference.py` — Local testing script (no server needed)
- `requirements.txt` — Python dependencies
- `README.md` — This file

## Notes

- Model is downloaded on first run (~500MB, cached locally after)
- Sentiment labels: `LABEL_0` (negative), `LABEL_1` (negative), `LABEL_2` (positive) — RoBERTa's original 3-class scheme, kept as-is
- Confidence scores are softmax probabilities (0-1)
- No fine-tuning was performed; this is inference-only using a pretrained model

## Run Locally

```bash
git clone https://github.com/samiyatanveer/roman-urdu-sentiment
cd roman-urdu-sentiment
pip install -r requirements.txt
streamlit run app.py
```

Open: http://localhost:8501
