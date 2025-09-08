from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os

from chatbot import predict_emotions  

# === API key ===
API_KEY = "my_secret_key_605"


# === FastAPI app ===
app = FastAPI(title="Emotion Recognition API")

# === Request schema ===
class TextInput(BaseModel):
    text: str
    threshold: float = 0.8
    use_gnn: bool = True

@app.post("/predict")
async def predict(input_data: TextInput, request: Request):
    # Check API key
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        predictions = predict_emotions(
            input_data.text,
            threshold=input_data.threshold,
            api_key=API_KEY,
            use_gnn=input_data.use_gnn
        )
        return {"text": input_data.text, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
