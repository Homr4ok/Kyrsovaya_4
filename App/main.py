import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import nltk
from rutermextract import TermExtractor
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
app = FastAPI()
term_extractor = TermExtractor()

def text_from_video():
    model = whisper.load_model("turbo")
    text = model.transcribe("input.mp3").get("text")
    sia = SentimentIntensityAnalyzer()
    res = sia.polarity_scores(text)
    max_val_key = max(res, key=res.get)
    score = "Тональность текста: "
    if max_val_key == "neu":
        score += "Нейтральный"
    elif max_val_key == "neg":
        score += "Негативный"
    elif max_val_key == "pos":
        score += "Позитивный"

    key_words = "Ключевые слова:"
    for term in term_extractor(text, limit=3, nested=True):
        key_words += " " + term.normalized

    os.remove("input.mp3")
    return score + "\n" + key_words + "\n" + text

@app.get('/')
def get_root():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    with open("input.mp3", "wb") as f:
        content = await file.read()
        f.write(content)

    return  JSONResponse(content={"text": text_from_video()})

