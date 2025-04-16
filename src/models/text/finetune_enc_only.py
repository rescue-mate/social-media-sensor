import argparse
import json
import os
from typing import List

from transformers import AutoTokenizer
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, label_mapping):
        """
        Args:
            texts (list): List of text examples.
            labels (list): List of text labels (e.g., "positive", "neutral").
            tokenizer (Tokenizer): Hugging Face tokenizer.
            label_mapping : a function to map from text labels to numeric values.
        """
        self.texts = [i['text'] for i in data]
        self.labels = [i['label'] for i in data]
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


def extract(filename):
    out = list()
    with open(filename, 'r') as file:
        data = json.load(file)
        for sample in data:
            out.append({'label':[sample['class_label']], 'text':sample['german_translation']})
    print(f'{len(out)} in total ')
    return out


def compute_metrics_binary(eval_pred):
    predictions, references = eval_pred

    # Apply sigmoid to logits to convert to probabilities
    predictions = 1 / (1 + np.exp(-predictions))

    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
    threshold = 0.5
    binary_predictions = (predictions >= threshold).astype(int)

    # Compute evaluation metrics
    f1 = f1_score(references, binary_predictions, average="macro")  # F1-score
    hamming = hamming_loss(references, binary_predictions)  # Hamming loss
    accuracy = accuracy_score(references, binary_predictions)  # Overall accuracy (strict)

    # print(classification_report(references, binary_predictions))
    return {
        "f1_macro": f1,
        "hamming_loss": hamming,
        "accuracy": accuracy,
    }


def compute_metrics(eval_pred):
    recall_metric = evaluate.load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return recall_metric.compute(predictions=predictions, references=labels)



def main(train_paths: List[str], dev_paths: List[str], test_paths: List[str],
         model_checkpoint: str ="google-bert/bert-base-german-cased", batch_size: int = 32,
         num_epochs: int = 8):

    os.environ["WANDB_PROJECT"] = "rescue_mate"

    train = []
    found_labels = []
    for path in train_paths:
        data = extract(path)
        train += data
        found_labels.append(sorted({i['label'][0] for i in data}))

    dev = []
    for path in dev_paths:
        dev += extract(path)

    test = []
    for path in test_paths:
        test += extract(path)

    label_mapping = {
        'affected_individual':'affected_individual',
        'caution_and_advice':'caution_and_advice',
        'displaced_and_evacuations':'displaced_and_evacuations',
        'donation_and_volunteering':'donation_and_volunteering',
        'infrastructure_and_utilities_damage':'infrastructure_and_utilities_damage',
        'injured_or_dead_people':'injured_or_dead_people',
        'missing_and_found_people':'missing_and_found_people',
        'not_humanitarian':'not_humanitarian',
        'requests_or_needs':'requests_or_needs',
        'response_efforts':'response_efforts',
        'sympathy_and_support':'sympathy_and_support',
        'displaced_people_and_evacuations':'displaced_and_evacuations',
        'infrastructure_and_utility_damage':'infrastructure_and_utilities_damage',
        'missing_or_found_people':'missing_and_found_people',
        'requests_or_urgent_needs':'requests_or_needs',
        'rescue_volunteering_or_donation_effort':'donation_and_volunteering',
        'other_relevant_information':'caution_and_advice',
    }

    for i in train:
        i['label'] = [label_mapping[x] for x in i['label']]
    for i in dev:
        i['label'] = [label_mapping[x] for x in i['label']]
    for i in test:
        i['label'] = [label_mapping[x] for x in i['label']]

    labels = [i['label'] for i in train]

    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    num_labels = len(mlb.classes_)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding="max_length", truncation=True)
    train = TextDataset(train, tokenizer, mlb)
    test = TextDataset(test, tokenizer, mlb)
    dev = TextDataset(dev, tokenizer, mlb)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, problem_type="multi_label_classification")

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-binary",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to="wandb",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=dev,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_binary,
    )

    trainer.train()

    best_ckpt_path = trainer.state.best_model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(best_ckpt_path)
    trainer.model = model.to(trainer.model.device)
    predictions = trainer.predict(test)
    trainer.log_metrics("test", predictions.metrics)
    # Log best model
    trainer.log({"best_model": best_ckpt_path})



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_paths",nargs='+', type=str, default=["data/event_data/cbd/translated_train.json", "data/event_data/HumAID/translated_train.json"])
    argparser.add_argument("--dev_paths",nargs='+', type=str, default=["data/event_data/cbd/translated_dev.json", "data/event_data/HumAID/translated_dev.json"])
    argparser.add_argument("--test_paths",nargs='+', type=str, default=["data/event_data/cbd/translated_test.json", "data/event_data/HumAID/translated_test.json"])

    args = argparser.parse_args()

    main(args.train_paths, args.dev_paths, args.test_paths)