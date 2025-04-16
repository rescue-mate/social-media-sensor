import pickle
from typing import List

import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
from transformers import AutoModelForImageClassification, AutoImageProcessor


class EventImageClassifier:
    def __init__(self, model_path: str):
        self.label2id = {'infrastructure_and_utility_damage': 0, 'rescue_volunteering_or_donation_effort': 1, 'not_humanitarian': 2, 'affected_injured_or_dead_people': 3}
        self.id2label = {0: 'infrastructure_and_utility_damage', 1: 'rescue_volunteering_or_donation_effort', 2: 'not_humanitarian', 3: 'affected_injured_or_dead_people'}
        self.model = AutoModelForImageClassification.from_pretrained(
            model_path,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="multi_label_classification",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running inference on {self.device}")
        self.model.to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        self.transforms = self._init_transform_images()

    def _init_transform_images(self):
        normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)
        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (self.image_processor.size["height"], self.image_processor.size["width"])
        )
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
        return _transforms
    def transform_images(self, images: List[Image]):
        pixel_values = [self.transforms(image.convert("RGB")) for image in images]
        return pixel_values
    def predict(self, images: List[Image], batch_size: int = 32):
        transformed_images = self.transform_images(images)
        predictions = []
        for batch_start in range(0, len(transformed_images), batch_size):
            batch = transformed_images[batch_start:batch_start + batch_size]
            batch = torch.stack(batch)
            with torch.no_grad():
                outputs = self.model(batch.to(self.device))
            # Change to multi-label classification in the future
            predicted_classes = outputs.logits.argmax(dim=-1).tolist()
            predicted_labels = [self.id2label[i] for i in predicted_classes]
            for label in predicted_labels:
                predictions.append({"predicted_labels": [label]})
        return predictions
