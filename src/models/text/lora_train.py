import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login
from torch.utils.data import Dataset as TorchDataset
import bitsandbytes as bnb
from transformers import EarlyStoppingCallback

def extract_data(filename):
    """
    Extract data from a JSON file and return it in the required format.
    """
    out = []
    with open(filename, 'r') as file:
        data = json.load(file)
        for sample in data:
            out.append({'label': [sample['post']['class_label']], 'text': sample['response']})
    print(f'{len(out)} samples extracted from {filename}')
    return out

class LoraDataset(TorchDataset):
    def __init__(self, data, tokenizer, mlb, test=False):
        """
        A dataset class to tokenize data for training and evaluation.
        """
        self.tokenizer = tokenizer
        self.texts = [self.generate_test_prompt(i['text'], mlb) if test else self.generate_prompt(i['text'], i['label'], mlb) for i in data]
        self.labels = [i['label'] for i in data]

    def generate_prompt(self, text, label, mlb):
        messages = [
            {"role": "system",
             "content": f"Classify the text into {', '.join(mlb.classes_)}, and return the answer as the corresponding natural disaster label."},
            {"role": "user", "content": f"Text: {text}."},
            {"role": "assistant", "content": f"label: {label}."}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def generate_test_prompt(self, text, mlb):
        messages = [
            {"role": "system",
             "content": f'Classify the text into {", ".join(mlb.classes_)}, and return the answer as the corresponding natural disaster label.'},
            {"role": "user", "content": f'text: {text}.'},
            {"role": "assistant", "content": f'label:'}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return item


def find_all_linear_names(model):
    """
    Find all linear module names in the model for LoRA tuning.
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, padding="max_length", truncation=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_data = extract_data(args.train_file)
    dev_data = extract_data(args.dev_file)

    labels = [i['label'] for i in train_data]
    target_names = [
        'affected_individual', 'caution_and_advice', 'displaced_and_evacuations',
        'donation_and_volunteering', 'infrastructure_and_utilities_damage',
        'injured_or_dead_people', 'missing_and_found_people', 'not_humanitarian',
        'requests_or_needs', 'response_efforts', 'sympathy_and_support'
    ]
    mlb = MultiLabelBinarizer(classes=target_names)
    mlb.fit(labels)

    train_dataset = LoraDataset(train_data, tokenizer, mlb, test=False)
    dev_dataset = LoraDataset(dev_data, tokenizer, mlb, test=False)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        device_map="auto",
        torch_dtype="float16",
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )

    training_args = TrainingArguments(
        run_name=args.run_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.0
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=512,
        packing=False,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with LoRA tuning.")
    parser.add_argument("--model_checkpoint", type=str, help="Model checkpoint name or path.", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--train_file", type=str, help="Path to the training dataset.", default="/storage/rescue_mate/translated_posts_crisis_consolidated_humanitarian_filtered_lang_en_train.json")
    parser.add_argument("--dev_file", type=str, help="Path to the validation dataset.", default="/storage/rescue_mate/translated_posts_crisis_consolidated_humanitarian_filtered_lang_en_dev.json")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--run_name", type=str, default="lora-crisisnlp", help="Run name for logging.")
    args = parser.parse_args()

    main(args)
