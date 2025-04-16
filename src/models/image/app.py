from io import BytesIO
from typing import List

from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File

from src.models.image.config import Settings
from src.models.image.image_classifier import EventImageClassifier

# Define the model and tokenizer as global variables
classifier: EventImageClassifier = None

settings = Settings()
app = FastAPI()

@app.on_event("startup")
async def load_model():
    """
    Initialize the model and tokenizer during app startup.
    """
    global classifier
    classifier = EventImageClassifier(settings.model_path)

    print("Initialized!")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Endpoint to predict the class of the given text.
    """
    global classifier
    if classifier is None:
        raise HTTPException(status_code=500, detail="Not initialized!")

    # Transform the image to a PIL Image
    try:
        image_data = await image.read()  # Read the file data
        pil_image = Image.open(BytesIO(image_data))  # Convert to PIL Image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format!") from e

    # Perform inference
    prediction = classifier.predict([pil_image])[0]


    return prediction

@app.post("/predict_batch")
async def predict_batch(images: List[UploadFile] = File(...)):
    """
    Endpoint to predict the class of the given text.
    """
    global classifier
    if classifier is None:
        raise HTTPException(status_code=500, detail="Not initialized!")

    # Transform the image to a PIL Image
    pil_images = []
    for image in images:
        try:
            image_data = await image.read()  # Read the file data
            pil_image = Image.open(BytesIO(image_data))  # Convert to PIL Image
            pil_images.append(pil_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image format!") from e

    # Perform inference
    predictions = classifier.predict(pil_images)

    return predictions

if __name__ == "__main__":
    # Start the FastAPI app
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
