import dataset
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('models/final_tokenizer_assets')
model = AutoModelForSequenceClassification.from_pretrained("models/saved_model.pt").to(device)
model.eval()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(text: TextInput):
    text = dataset.clean_data(text.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()

    return {"text": text.text, "cleaned_text": text, "prediction": int(pred_class)}

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}