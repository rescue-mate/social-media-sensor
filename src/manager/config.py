from typing import List

from pydantic import BaseModel, Field

from src.models.image.config import ImageResponse
from src.models.text.config import TextResponse


class SocialMediaPostResponse(BaseModel):
    text_prediction: TextResponse = Field(..., description="The prediction for the text.")
    image_predictions: List[ImageResponse] = Field(..., description="The predictions for the images.")
    collective_image_prediction: List[str] = Field(..., description="The collective predictions for the images.")
