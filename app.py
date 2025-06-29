from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import pytesseract
from PIL import Image
import io
import uvicorn


app = FastAPI(
    title="Cyberbullying Detection API",
    description="API that extracts text from images and detects cyberbullying content.",
    version="1.0"
)

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", return_all_scores=True)

def get_severity(scores):
    """
    Converts model scores to severity levels.
    """
    bullying_labels = ["toxic", "insult", "severe_toxic", "threat", "obscene"]
    
    
    bullying_scores = [scores[label] for label in bullying_labels if label in scores]

    
    if not bullying_scores:
        return "safe"

    max_score = max(bullying_scores)

    if max_score < 0.3:
        return "safe"
    elif max_score < 0.5:
        return "mild"
    elif max_score < 0.7:
        return "insult"
    else:
        return "severe"


class TextInput(BaseModel):
    text: str

@app.post("/extract_text/", summary="Extract text from an image", tags=["OCR"])
async def extract_text(image: UploadFile = File(...)):
    """
    Extracts text from an uploaded image using OCR (Tesseract).
    - **image**: Image file containing text.

    **Returns:**
    - `extracted_text`: The text extracted from the image.
    """
    try:
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))

        extracted_text = pytesseract.image_to_string(image)

        return {"extracted_text": extracted_text}

    except Exception as e:
        return {"error": str(e)}


@app.post("/detect/", summary="Detect cyberbullying content in text", tags=["Detection"])
async def detect_cyberbullying(input: TextInput):
    """
    Detects cyberbullying content in the given text.
    - **text**: Input text to analyze.

    **Returns:**
    - `text`: Original text.
    - `severity`: Severity level (safe, mild, insult, severe).
    - `scores`: Confidence scores for each category.
    """
    result = classifier(input.text)

    
    scores = {res["label"]: res["score"] for res in result[0]}

    severity = get_severity(scores)

    return {"text": input.text, "severity": severity, "scores": scores}

#server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
# see uvicorn app:app --host 0.0.0.0 --port 8000 --reload
