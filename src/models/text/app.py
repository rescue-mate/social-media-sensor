from fastapi import FastAPI, HTTPException

from src.models.text.base_inference import TextClassificationModel
from src.models.text.config import TextRequest, BatchTextRequest, Settings
from src.models.text.config import load_model as load_model_

# Define the model and tokenizer as global variables
classifier: TextClassificationModel = None

settings = Settings()
app = FastAPI()

@app.on_event("startup")
async def load_model():
    """
    Initialize the model and tokenizer during app startup.
    """
    global classifier
    classifier = load_model_(settings.text_classifier, settings.model_path)


@app.post("/predict")
async def predict(request: TextRequest):
    """
    Endpoint to predict the class of the given text.
    """
    global classifier
    if classifier is None:
        raise HTTPException(status_code=500, detail="Not initialized!")


    # Perform inference
    prediction = classifier.predict([request.text])[0]


    return {"text": request.text, **prediction}

@app.post("/predict_batch")
async def predict_batch(request: BatchTextRequest):
    """
    Endpoint to predict the class of the given text.
    """
    global classifier
    if classifier is None:
        raise HTTPException(status_code=500, detail="Not initialized!")

    # Perform inference
    predictions = classifier.predict(request.texts)
    return [{"text": text, **prediction} for text, prediction in zip(request.texts, predictions)]

if __name__ == "__main__":
    # Start the FastAPI app
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
