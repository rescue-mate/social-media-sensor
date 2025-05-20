import logging
from collections import defaultdict
from typing import List

import pydantic_settings

from src.models.combined.few_shot_multimodal_inference import InContextMMClassificationModel
from src.models.geo_linker.geo_linker import GeoLinker
from src.models.image.image_classifier import EventImageClassifier
import concurrent.futures
from src.models.text.few_shot_inference import InContextClassificationModel
from src.models.utils import MMWrapper, VLMWrapper, OAWrapper
logger =  logging.getLogger(__name__)


def merge_image_predictions(image_predictions: list):
    """
    Function to merge the predictions from text and image classifiers.
    """
    predictions = set()
    for image in image_predictions:
        predicted_labels = image.get("predicted_labels", [])
        predictions.update(predicted_labels)
    return list(predictions)


class EventSensor:
    def __init__(self, settings: pydantic_settings.BaseSettings):
        llm = OAWrapper(settings.llm_address, settings.model_name)
        if not settings.multi_modal:
            self.text_classifier = InContextClassificationModel(llm)
            if settings.image_model.model_path is not None:
                self.image_classifier = EventImageClassifier(settings.image_model.model_path)
            else:
                self.image_classifier = None
            self.combined_classifier = None
        else:
            self.combined_classifier = InContextMMClassificationModel(llm)
            self.text_classifier = None
            self.image_classifier = None
        self.always_geo_link = settings.always_geo_link
        self.geo_linker = GeoLinker(llm, num_examples_per_class=settings.geo_linker.num_examples_per_class,
                                    batch_size=settings.geo_linker.batch_size,
                                    photon_url=settings.geo_linker.photon_url,
                                    nominatim_url=settings.geo_linker.nominatim_url,
                                    default_city=settings.geo_linker.default_city,
                                    default_country=settings.geo_linker.default_country,
                                    default_district=settings.geo_linker.default_district,
                                    num_candidates=settings.geo_linker.num_candidates,
                                    do_llm_filtering=settings.geo_linker.do_llm_filtering,
                                    few_shots=settings.geo_linker.few_shots,
                                    do_collective_linking=settings.geo_linker.do_collective_linking)

    def do_event_classification(self, messages):
        logger.info("Start event classification")
        if self.text_classifier is not None and self.image_classifier is not None:
            texts = []
            images = []
            message_to_images = defaultdict(int)
            message_to_text = {}
            for idx, message in enumerate(messages):
                if message.get("text") is not None:
                    message_to_text[idx] = len(texts)
                    texts.append(message["text"])
                for image in message.get("images", []):
                    message_to_images[idx].append(len(images))
                    images.append(image)

            text_predictions = self.text_classifier.predict(texts)
            image_predictions = self.image_classifier.predict(images)
            responses = []
            for idx in range(len(messages)):
                if idx in message_to_text:
                    text_prediction = text_predictions[idx]
                else:
                    text_prediction = None
                image_predictions_ = [image_predictions[i] for i in message_to_images[idx]]
                merged_predictions = merge_image_predictions(image_predictions_)
                response = {
                    "prediction": text_prediction,
                    "collective_image_prediction": merged_predictions,
                    "image_predictions": image_predictions_,
                    "text": texts[message_to_text[idx]] if idx in message_to_text else None
                }
                responses.append(response)

        else:
            for message in messages:
                if message.get("text") is None:
                    # Add text prompt for image only prediction
                    message["text"] = "Schau dir das an."
            response = self.combined_classifier.predict(messages)
            responses = [
                {
                    "prediction": item["predicted_labels"] if item["predicted_labels"] else ["not_humanitarian"],
                    "collective_image_prediction": item["predicted_labels"],
                    "image_predictions": [[]] * len(message.get("images", [])),
                    "text": message["text"]
                }
                for item, message in zip(response, messages)
            ]
        logger.info("Event classification done.")
        return responses

    def predict(self, messages: List[dict]):
        # Each message consists of a text and/or image

        texts: List[str] = [x["text"] for x in messages]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_classify = executor.submit(self.do_event_classification, messages)
            future_geolink = executor.submit(self.geo_linker.link_texts, texts)

            results = future_classify.result()
            geo_links = future_geolink.result()

        logger.info("Both classification and geo-linking complete.")

        for result, geo_link in zip(results, geo_links):
            if result["text"].strip():
                result["geo_linked_entities"] = geo_link
            else:
                result["geo_linked_entities"] = []

        logger.info("Done")

        return results




