import asyncio
import json
import os
import sys
import uuid
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

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),  # fallback to INFO if invalid
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger =  logging.getLogger(__name__)

# Define the model and tokenizer as global variables
event_sensor: EventSensor = None
background_task: asyncio.Task = None

app = FastAPI()

class Settings(BaseSettings):
    multi_modal: bool = True
    image_model: Optional[ImageSettings] = None
    llm_address: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    geo_linker: GeoLinkerSettings = GeoLinkerSettings()
    download_images: bool = False
    always_geo_link: bool = True
    target_server: str = "https://rescue-mate.uhh-net.de"
    target_server_access_token: str = "your_access_token_here"
    target_server_dataset: str = "social_media"

    class Config:
        env_nested_delimiter = '__'

settings = Settings()
authorization_headers = {"Authorization": f"Bearer {settings.target_server_access_token}"}
# Create the PostCollector instance
collector = SocialMediaCollector()

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
    global collector

    # Start the collector (run the WebSocket connection in the background)
    collector.start()


    while True:
        data = retrieve_and_classify_posts(collector)

        await websocket.send_json(data)
        await asyncio.sleep(1)  # Send data every second

def send_data_to_target_server(data):
    if data:
        for elem in data:
            # Generate UUID for each post
            elem["post_id"] = str(uuid.uuid4())
        data_as_json = json.dumps(data)
        response = requests.post(f"{settings.target_server}/datasets/{settings.target_server_dataset}", headers=authorization_headers,
                      files={
                        "data": ("data.json", data_as_json.encode("utf-8"), "application/json")
                        })
        if response.status_code != 200:
            logger.warning(f"Failed to send data to target server: {response.status_code} - {response.text}")
        else:
            logger.info(f"Data sent to target server: {settings.target_server}/datasets/{settings.target_server_dataset}/data")
async def send_data_periodically():
    global collector

    while True:
        data = retrieve_and_classify_posts(collector)

        send_data_to_target_server(data)
        await asyncio.sleep(1)  # Send data every second
def setup_platform_feed():
    # Check if target server is reachable
    response = requests.get(settings.target_server + "/datasets", headers=authorization_headers)
    if response.status_code != 200:
        logger.warning(f"Target server is not reachable: {settings.target_server}")
        return


    logger.info(f"Target server is reachable: {settings.target_server}")
    # Check if dataset exists
    response = requests.get(f"{settings.target_server}/datasets/{settings.target_server_dataset}", headers=authorization_headers)
    if response.status_code == 404:
        logger.info(f"Dataset does not exist: {settings.target_server_dataset}. Creating it.")
        with open("data/rml.ttl", "rb") as rml_file:
            response = requests.post(
                f"{settings.target_server}/datasets",
                data={"name": settings.target_server_dataset},
                files={"rml": rml_file},
                headers=authorization_headers
            )
    if response.status_code != 200:
        logger.warning(f"Failed setup: {settings.target_server_dataset}.")
        return



    return asyncio.create_task(send_data_periodically())

@app.on_event("startup")
async def load_model():
    """
    Initialize the model and tokenizer during app startup.
    """
    global event_sensor

    event_sensor = EventSensor(settings)

async def start_background_task():
    global background_task, collector
    if background_task is None or background_task.done():
        logger.info("Starting background task.")
        collector.start()
        background_task = setup_platform_feed()
    else:
        logger.info("Background task is already running.")

async def stop_background_task():
    global background_task
    global collector
    if background_task and not background_task.done():
        logger.info("Stopping background task.")
        collector.stop()
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            logger.info("Background task cancelled successfully.")
    else:
        logger.info("No background task running.")

@app.post("/start-feed")
async def start_feed():
    await start_background_task()
    return {"status": "started"}

@app.post("/stop-feed")
async def stop_feed():
    await stop_background_task()
    return {"status": "stopped"}

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
    uvicorn.run("app:app", host="0.0.0.0", port=8060, reload=False)
