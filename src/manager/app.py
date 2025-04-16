import asyncio
import sys
from io import BytesIO
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.params import File, Form
from PIL import Image
from pydantic_settings import BaseSettings
from starlette.websockets import WebSocket

from src.manager.config import SocialMediaPostResponse
from src.manager.social_media_collector import SocialMediaCollector
from src.models.event_sensor import EventSensor
from src.models.geo_linker.config import GeoLinkerSettings
from src.models.image.config import ImageResponse
from src.models.text.config import TextRequest, BatchTextRequest, TextResponse
from src.models.image.config import Settings as ImageSettings
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger =  logging.getLogger(__name__)

# Define the model and tokenizer as global variables
event_sensor: EventSensor = None

app = FastAPI()

class Settings(BaseSettings):
    multi_modal: bool = True
    image_model: Optional[ImageSettings] = None
    llm_address: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    geo_linker: GeoLinkerSettings = GeoLinkerSettings()
    download_images: bool = False
    always_geo_link: bool = True

    class Config:
        env_nested_delimiter = '__'

settings = Settings()
logger.info("Settings loaded from environment variables.")
logger.info(f"Settings: {settings}")

def retrieve_and_classify_posts(collector):
    """
    Function to retrieve and classify posts.
    """
    global event_sensor

    posts = collector.get_all_posts()

    if posts:
        new_posts = []
        for post in posts:
            image_urls = []
            for image in post.get('images', []):
                image_urls.append(image['url'])

            # Download images
            images = []
            if settings.download_images:
                for image in image_urls:
                    try:
                        image = Image.open(requests.get(image, stream=True).raw)
                        images.append(image)
                    except Exception as e:
                        logger.info(f"Error downloading image: {e}")
                        continue

            new_post = {
                "text": post['text'],
                "images": images
            }
            new_posts.append(new_post)


        classified_posts = event_sensor.predict(new_posts)
        for post, classified_post in zip(posts, classified_posts):
            classified_post.update(post)
    else:
        classified_posts = []

    return classified_posts



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Create the PostCollector instance
    collector = SocialMediaCollector()

    # Start the collector (run the WebSocket connection in the background)
    collector.start()


    while True:
        data = retrieve_and_classify_posts(collector)

        await websocket.send_json(data)
        await asyncio.sleep(1)  # Send data every second


@app.on_event("startup")
async def load_model():
    """
    Initialize the model and tokenizer during app startup.
    """
    global event_sensor

    event_sensor = EventSensor(settings)

@app.post("/predict_text", summary="Predict the classes of the given text.")
async def predict_text(request: TextRequest) -> TextResponse:
    """
    Endpoint to predict the class of the given text.
    - **text**: The text to predict the class for.
    """
    global event_sensor
    if event_sensor is None:
        raise HTTPException(status_code=500, detail="Not initialized!")


    prediction = event_sensor.predict([{"text": request.text}])[0]

    response = TextResponse(text=request.text, predicted_labels=prediction["prediction"],
                            geo_linked_entities=prediction["geo_linked_entities"])

    return response

@app.post("/predict_text_batch", summary="Predict the classes of the given texts.")
async def predict_text_batch(request: BatchTextRequest) -> List[TextResponse]:
    """
    Endpoint to predict the class of the given texts.
    - **texts**: The texts to predict the class for.
    """
    global event_sensor
    if event_sensor is None:
        raise HTTPException(status_code=500, detail="Not initialized!")

    predictions = event_sensor.predict([{"text": text} for text in request.texts])

    text_responses = []
    for text, prediction in zip(request.texts, predictions):
        text_responses.append(TextResponse(text=text, predicted_labels=prediction["prediction"],
                                           geo_linked_entities=prediction["geo_linked_entities"]))

    return text_responses

async def process_image(image: UploadFile) -> Image:
    # Transform the image to a PIL Image
    try:
        image_data = await image.read()  # Read the file data
        pil_image = Image.open(BytesIO(image_data))  # Convert to PIL Image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format!") from e
    return pil_image


@app.post("/predict_image", summary="Predict the classes of the given image.")
async def predict_image(image: UploadFile = File(...)) -> ImageResponse:
    """
    Endpoint to predict the class of the given text.
    - **image**: The image to predict the class for.
    """
    if event_sensor is None:
        raise HTTPException(status_code=500, detail="Not initialized!")

    pil_image = await process_image(image)
    # Perform inference
    prediction = event_sensor.predict([{"images": [pil_image]}])[0]

    response = ImageResponse(image_name=image.filename, predicted_labels=prediction["collective_image_prediction"])

    return response

@app.post("/predict_image_batch", summary="Predict the classes of the given images.")
async def predict_image_batch(images: List[UploadFile] = File(...)) -> List[ImageResponse]:
    """
    Endpoint to predict the class of the given text.
    - **images**: The images to predict the classes for each image.
    """
    global event_sensor
    if event_sensor is None:
        raise HTTPException(status_code=500, detail="Not initialized!")

    # Transform the image to a PIL Image
    pil_images = []
    for image in images:
        pil_image = await process_image(image)
        pil_images.append(pil_image)

    # Perform inference
    predictions = event_sensor.predict([{"images": [pil_image]} for pil_image in pil_images])

    responses = []
    for image, prediction in zip(images, predictions):
        responses.append(ImageResponse(image_name=image.filename, predicted_labels=prediction["collective_image_prediction"]))

    return responses

@app.post("/predict_social_media_post", summary="Predict the classes of the given text and images.")
async def predict_social_media_post(text: str = Form(...), images: List[UploadFile] = File(...)) -> SocialMediaPostResponse:
    """
    Function to predict the classes of the given text and images.
    It returns the predictions from both the text and image classifiers.
    - **text**: The text to predict the class for.
    - **images**: The images to predict the classes for.
    """
    global event_sensor
    if event_sensor is None:
        raise HTTPException(status_code=500, detail="Not initialized!")

    processed_images = [await process_image(image) for image in images]
    response = event_sensor.predict([{"text": text, "images": processed_images}])[0]
    text_prediction = response["prediction"]
    collective_image_prediction = response["collective_image_prediction"]
    image_predictions = response["image_predictions"]

    response = SocialMediaPostResponse(text_prediction=TextResponse(text=text, predicted_labels=text_prediction,
                                                                    geo_linked_entities=response["geo_linked_entities"]),
                                       image_predictions=[
                                           ImageResponse(image_name=image.filename, predicted_labels=image_prediction) for image, image_prediction in
                                           zip(images, image_predictions)],
                                         collective_image_prediction=collective_image_prediction)

    return response


if __name__ == "__main__":
    # Start the FastAPI app
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
