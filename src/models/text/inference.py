from typing import List

from transformers import TrainingArguments
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
import numpy as np
from sklearn.metrics import classification_report
import json
import os
import argparse

from src.models.text.base_inference import TextClassificationModel

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Inference will run on CPU.")

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, label_mapping):
        """
        Args:
            data (list): List of dictionaries with keys "text" and "label".
            tokenizer (Tokenizer): Hugging Face tokenizer.
            label_mapping : a function to map from text labels to numeric values.
        """
        self.texts = [i['text'] for i in data]
        self.labels = [i['label'] for i in data]
        self.data = data
        self.labels = label_mapping.transform(self.labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",  # Pad to max length
            truncation=True,       # Truncate to max length
            max_length=512,        # Set max sequence length
            return_tensors="pt"    # Return PyTorch tensors
        )

        # Add the encoded label
        item = {key: val.squeeze(0) for key, val in encoding.items()}  
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

class EncoderTextClassificationModel(TextClassificationModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding="max_length", truncation=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)


    def prepare_dataset(self, file_path: str, include_label: bool = True):
        data = []
        with open(file_path, 'r') as file:
            loaded_data = json.load(file)
            for sample in loaded_data:
                if include_label:
                    data.append({'label':[sample.get('label')], 'text':sample['text']})
                else:
                    data.append({'label':["injured_or_dead_people"], 'text':sample['text']})
        return TextDataset(data, self.tokenizer, self.mlb)

    def predict(self, texts: List[str], batch_size: int = 32):
        predictions = []
        index_to_item = {index: item for index, item in enumerate(self.target_names)}
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            tokenized = self.tokenizer(batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**tokenized)
            logits = outputs.logits
            probabiltiies = torch.sigmoid(logits)
            for probabiltiies_ in probabiltiies:
                predicted_classes = torch.nonzero(probabiltiies_ > 0.5).flatten().tolist()
                predicted_labels = [index_to_item[i] for i in predicted_classes]
                predictions.append({
                    "predicted_labels": predicted_labels,
                    "probabilities": {target_name: probabiltiies_[index].item() for index, target_name in enumerate(self.target_names)}
                })
        return predictions

    def generate_annotations(self, file_path: str, include_label: bool, output_path: str):
        dataset = self.prepare_dataset(file_path)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./results",
                report_to="none",
            )
        )
        predictions = trainer.predict(dataset)
        predicted_labels = predictions.predictions.argmax(axis=-1)
        predicted_logit = predictions.predictions

        new = list()

        if include_label:
            item_to_index = {item: index for index, item in enumerate(self.target_names)}
            mapped_labels = [item_to_index[x[0]] for x in dataset.labels]

            report = classification_report(y_true=mapped_labels, y_pred=predicted_labels,target_names= self.target_names, output_dict= True, zero_division= True)
            new.append({"report":report})
            json.dumps(report, indent=4)

        index_to_item = {index: item for index, item in enumerate(self.target_names)}
        predicted_labels = [index_to_item[i] for i in predicted_labels]

        for predicted_label, data, logit in zip(predicted_labels, dataset.data, predicted_logit):
            data["predicted_label"] = predicted_label
            data["logit"] = logit.tolist() if isinstance(logit, np.ndarray) else logit 
            new.append(data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new, f, ensure_ascii=False, indent=4)
    
        print(f"Saving file to: {os.path.abspath(output_path)}")
        return predicted_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Classification Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the predictions, in a json file.")
    parser.add_argument("--include_label", action="store_true", help="whether there is label in the input file")

    args = parser.parse_args()

    classifier = EncoderTextClassificationModel(args.model_path)
    predictions = classifier.generate_annotations(file_path=args.file_path, include_label=args.include_label, output_path=args.output_path)
    