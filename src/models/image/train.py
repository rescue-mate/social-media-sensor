import csv

from sklearn.metrics import f1_score, hamming_loss, accuracy_score, classification_report
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, DefaultDataCollator

import PIL.Image as Image
import numpy as np
import evaluate

from transformers import AutoImageProcessor

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def load_datasets(image_processor):
    from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    labels = set()
    train_dataset = []
    for elem in csv.DictReader(open("data_humanitarian/train.csv", "r", encoding="utf-8"), delimiter="\t"):
        train_dataset.append({"pixel_values": _transforms(Image.open(elem["filepath"]).convert("RGB")), "label": elem["title"]})
        labels.add(elem["title"])

    eval_dataset = []
    for elem in csv.DictReader(open("data_humanitarian/dev.csv", "r", encoding="utf-8"), delimiter="\t"):
        eval_dataset.append({"pixel_values": _transforms(Image.open(elem["filepath"]).convert("RGB")), "label": elem["title"]})
        labels.add(elem["title"])

    labels = list(labels)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    for data in train_dataset:
        label = np.zeros(len(labels))
        label[label2id[data["label"]]] = 1
        data["label"] = label
    for data in eval_dataset:
        label = np.zeros(len(labels))
        label[label2id[data["label"]]] = 1
        data["label"] = label

    return train_dataset, eval_dataset, id2label, label2id

def compute_metrics_binary(eval_pred):
    predictions, references = eval_pred

    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
    threshold = 0.5
    binary_predictions = (predictions >= threshold).astype(int)

    # Compute evaluation metrics
    f1 = f1_score(references, binary_predictions, average="macro")  # F1-score
    hamming = hamming_loss(references, binary_predictions)  # Hamming loss
    accuracy = accuracy_score(references, binary_predictions)  # Overall accuracy (strict)

    cr=  classification_report(references, binary_predictions)
    return {
        "f1_macro": f1,
        "hamming_loss": hamming,
        "accuracy": accuracy,
        "classification_report": cr
    }

def main():
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    train_dataset, eval_dataset, id2label, label2id = load_datasets(image_processor)
    data_collator = DefaultDataCollator()

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="crisis",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        compute_metrics=compute_metrics_binary,
    )

    print(label2id)

    trainer.train()


if __name__ == "__main__":
    main()