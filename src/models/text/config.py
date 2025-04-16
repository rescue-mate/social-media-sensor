from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from src.models.text.few_shot_inference import InContextClassificationModel
from src.models.text.inference import EncoderTextClassificationModel
from src.models.text.lora_inference import DecoderTextClassificationModel
from src.models.utils import VLMWrapper, OAWrapper


class TextRequest(BaseModel):
    text: str = Field(..., description="The text to predict the class for.")

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="The texts to predict the class for.")

class TextResponse(BaseModel):
    text: str = Field(..., description="The text to predict the class for.")
    predicted_labels: List[str] = Field(..., description="The predicted labels for the text.")
    geo_linked_entities: List[dict] = Field(..., description="The linked entities for the text.")

class TextClassifier(Enum):
    encoder = "encoder"
    lora = "lora"
    in_context = "in_context"


class Settings(BaseSettings):
    model_path: str
    text_classifier: TextClassifier = TextClassifier.encoder


def load_model(text_classifier: TextClassifier, model_path, open_ai_address):
    """
    Initialize the model and tokenizer during app startup.
    """
    global classifier
    llm = None
    if text_classifier == TextClassifier.encoder:
        classifier = EncoderTextClassificationModel(model_path)
    elif text_classifier == TextClassifier.lora:
        classifier = DecoderTextClassificationModel(model_path)
    elif text_classifier == TextClassifier.in_context:
        if open_ai_address:
            llm = OAWrapper(model_path, open_ai_address)
        else:
            llm = VLMWrapper(model_path)
        classifier = InContextClassificationModel(llm)

    return classifier, llm
