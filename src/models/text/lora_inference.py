import argparse
from typing import List, Dict
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.models.text.base_inference import TextClassificationModel


def parse_args():
    parser = argparse.ArgumentParser(description='Event Classification')
    parser.add_argument('--model_path', type=str, default="/storage/rescue_mate/checkpoint-955")
    parser.add_argument('--input_file', type=str, default="/storage/rescue_mate/translated_posts_crisis_consolidated_humanitarian_filtered_lang_en_test.json")
    parser.add_argument('--output_file', type=str, default="results_lora.json")
    parser.add_argument('--include_label', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

class DecoderTextClassificationModel(TextClassificationModel):

    def __init__(self, model_path: str, batch_size: int = 32, max_new_tokens: int = 100):
        super().__init__()
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self.model_init()

    def predict(self, texts: List[str]) -> List[Dict]:
        prompts = [self.generate_test_prompt(text) for text in texts]
        predictions = self.generate_text_batch(prompts,
                                               enable_prog_bar=False)
        return [{"predicted_labels": prediction} for prediction in predictions]

    def model_init(self) -> tuple:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", padding_side='left', truncation=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype="float16",
            quantization_config=bnb_config, 
        )
        model.config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.eval()
        return model, tokenizer

    def generate_annotations(self, file_path: str, include_label: bool, output_path: str):
        data, texts, labels = self.prepare_data(file_path, include_label)
        generated_texts = self.generate_text_batch(texts)
        self.evaluate_save(labels, generated_texts, data, include_label, output_path)

    def generate_text_batch(self, input_text:list, enable_prog_bar: bool = True):
        all_generated_texts = []
        batch_range = range(0, len(input_text), self.batch_size)
        if enable_prog_bar:
            batch_range = tqdm(batch_range)
        for i in batch_range:
            batch_prompts = input_text[i:i + self.batch_size]

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)

            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [gen.split("label:")[-1].strip() for gen in generated_texts]
            matched_labels = [[label for label in self.target_names if label.lower() in output.lower()] for output in generated_texts]
            all_generated_texts.extend(matched_labels)
        return all_generated_texts
    def extract_data(self, filename: str, include_label: bool) -> List[Dict]:
        out = list()
        with open(filename, 'r') as file:
            data = json.load(file)
            for sample in data:
                if include_label:
                    label = [sample['post']['class_label']]
                else:
                    label = []
                out.append({'label': label, 'text': sample['text']})
        print(f'{len(out)} samples in total')
        return out
    
    def generate_test_prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": f'Classify the text into {", ".join(self.target_names)}, and return the answer as the corresponding natural disaster label.'},
            {"role": "user", "content": f'text: {text}.'}
        ]   
        return self.tokenizer.apply_chat_template(messages, tokenize=False)
    
    def prepare_data(self, test_file: str, include_label: bool) -> tuple:
        test_data = self.extract_data(test_file, include_label)
        test_text = [self.generate_test_prompt(i["text"]) for i in test_data]
        if include_label:
            test_label = [i["label"] for i in test_data]
        else:
            test_label = []

        return test_data, test_text, test_label
    
    def evaluate_save(self, y_true: List, y_pred: List, test_data, include_label: bool, output_path: str) -> None:
        results = {
            "report": None,
            "predictions": []
        }
        
        mapped_pred = list()
        for matched_labels in y_pred:

            if matched_labels:
                mapped_pred.append(matched_labels)  
            else:
                mapped_pred.append(["injured_or_dead_people"])
        
        if include_label:
            mapped_pred_array = self.mlb.transform(mapped_pred)
            y_true_array = self.mlb.transform(y_true)
            class_report = classification_report(y_true=y_true_array, y_pred=mapped_pred_array, target_names=self.target_names, output_dict=True, zero_division=True)
            print('\nClassification Report:')
            print(class_report)
            results["report"] = class_report

        for pred_label, sample in zip(mapped_pred, test_data):
            sample["predicted_label"] = pred_label
            results["predictions"].append(sample)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Results saved to: {output_path}")

    

def main():
    args = parse_args()

    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Training will run on CPU.")
    
    classifier = DecoderTextClassificationModel(args.model_path, args.batch_size, args.max_new_tokens)
    classifier.generate_annotations(args.input_file, args.include_label, args.output_file)



if __name__ == "__main__":
    main()