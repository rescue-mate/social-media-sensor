from typing import List

from fastapi import UploadFile, Form
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ImageResponse(BaseModel):
    """
    Response schema for the image classification endpoint.
    """
    image_name: str = Form(..., description="The name of the image.")
    predicted_labels: List[str] = Form(..., description="The predicted labels for the image.")



class Settings(BaseSettings):
    model_path: str
