from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Use pretrained sentiment model directly (no fine-tuning)
pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    result = pipe(request.text)
    return {
        "text": request.text,
        "sentiment": result[0]['label'],
        "confidence": float(result[0]['score'])
    }

@app.get("/")
def home():
    return {"message": "Roman Urdu Sentiment Classifier API - Pretrained Model"}