import abc
import base64
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, DynamicCache
from vllm import LLM, SamplingParams
logger =  logging.getLogger(__name__)


class LLMWrapper:
    def __init__(self, top_p = 0.95,
                 temperature = 0.8,
                 max_tokens = 256):
        self.llm = None
        self.tokenizer = None
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abc.abstractmethod
    def generate(self, messages: list):
        pass

class VLMWrapper(LLMWrapper):
    def __init__(self, model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__()
        self.llm = LLM(model=model_path, dtype=torch.bfloat16, quantization="bitsandbytes",
                  load_format="bitsandbytes", max_model_len=4028, enable_prefix_caching=True)
        self.tokenizer = self.llm.llm_engine.tokenizer.tokenizer
        self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)

    def generate(self, messages: list):
        prompt_token_ids = [self.tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]

        responses = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params)
        return [x.outputs[0].text for x in responses]


class MMWrapper(LLMWrapper):
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__()
        #self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #    model_path, device_map="auto",
        #    torch_dtype=torch.bfloat16,
        #    attn_implementation="flash_attention_2",
        #)
        self.llm = LLM(model=model_path, dtype=torch.bfloat16,
                       #quantization="bitsandbytes",
                       #load_format="bitsandbytes",
                       max_model_len=4096, limit_mm_per_prompt={"image": 3},
                       enable_prefix_caching=True)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        #self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.llm.llm_engine.tokenizer.tokenizer
        self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens)


        self.prompt_dictionary = {}

    def get_cache(self, common_message: str):
        if common_message not in self.prompt_dictionary:
            with torch.no_grad():
                inputs = self.processor(
                    text=[common_message],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.llm.device) for key, value in inputs.items()}
                prompt_cache = DynamicCache()
                prompt_cache = self.llm(**inputs, past_key_values=prompt_cache).past_key_values
                self.prompt_dictionary[common_message] = prompt_cache
        return copy.deepcopy(self.prompt_dictionary[common_message])

    def construct_prompt_cache(self, all_prompts, inputs):
        prompt_caches = []
        for input_, prompt in zip(inputs,all_prompts):
            truncated_prompt = prompt[:prompt.find("Examples:")]
            prompt_caches.append(self.get_cache(truncated_prompt))

        key_cache = []
        value_cache = []
        for prompt_cache in prompt_caches:
            key_cache.append(prompt_cache.key_cache)
            value_cache.append(prompt_cache.value_cache)
        new_key_cache = []
        for key_cache_list in zip(*key_cache):
            key_cache_list = pad_sequence([x.squeeze(0) for x in key_cache_list], batch_first=True, padding_value=0,
                                          padding_side="left")
            new_key_cache.append(key_cache_list)
        new_value_cache = []
        for value_cache_list in zip(*value_cache):
            value_cache_list = pad_sequence([x.squeeze(0) for x in value_cache_list], batch_first=True, padding_value=0,
                                            padding_side="left")
            new_value_cache.append(value_cache_list)

        prompt_cache = prompt_caches[0]
        prompt_cache.key_cache = new_key_cache
        prompt_cache.value_cache = new_value_cache
        return prompt_cache





    def generate(self, messages: list = None, batch_size=24):
        all_image_inputs = []
        for message in messages:
            image_inputs, video_inputs = process_vision_info([message])
            if image_inputs:
                all_image_inputs.append(image_inputs[:3])
            else:
                all_image_inputs.append(None)

        all_prompts = [self.processor.apply_chat_template(prompt, add_generation_prompt=True,
                                                          tokenize=False) for prompt in messages]
        inputs = []
        for prompt, image_input in zip(all_prompts, all_image_inputs):
            if image_input:
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image_input
                    },
                })
            else:
                inputs.append({
                    "prompt": prompt,
                })


        responses = self.llm.generate(inputs, sampling_params=self.sampling_params)

        return [x.outputs[0].text for x in responses]



class OAWrapper(LLMWrapper):
    def __init__(self, url: str = "http://localhost:8000/v1", model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__()
        self.client = OpenAI(
            base_url=url,
        )


        # Send post requests via requests as well for testing



        #Test if the model is available
        while True:
            try:
                response = requests.get(f"{url}/models",)

                available_models = [x["id"] for x in response.json()["data"]]

                logger.info(f"Available models: {available_models}")

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Tell everything about the Mount Everest!"}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                break
            except Exception as e:
                logger.exception(f"Model {model_name} is not available yet, retrying in 5 seconds...")
                time.sleep(5)


        self.model_path = model_name

        self.prompt_dictionary = {}
        self.request_pool = ThreadPoolExecutor(14)

    def send_request(self, x):
        responses = self.client.chat.completions.create(
            model=self.model_path,
            messages=x,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return responses
    def generate(self, messages: list = None, batch_size=24):
        # responses = self.llm.generate(inputs, sampling_params=self.sampling_params)
        all_responses = list(self.request_pool.map(self.send_request, messages))

        return [x.choices[0].message.content for x in all_responses]