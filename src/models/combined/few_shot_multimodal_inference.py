import argparse
import base64
import dataclasses
import itertools
import json
import random
from collections import defaultdict
from typing import List, Dict

import faiss
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from src.models.text.base_inference import TextClassificationModel

@dataclasses.dataclass
class IndexTuple:
    index: faiss.Index
    mapping: Dict[int, int]

class InContextMMClassificationModel(TextClassificationModel):

    def __init__(self, llm, balanced_few_shots: bool = True,
                 ignore_negative: bool = True,
                 num_examples_per_class: int = 2):
        super().__init__()

        self.llm = llm

        self.balanced_few_shots = balanced_few_shots
        self.ignore_negative = ignore_negative
        self.num_examples_per_class = num_examples_per_class

        self.prompt = open("data/prompt.txt").read()
        self.mm_prompt = open("data/mm_prompt.txt").read()
        self.label_mapping = {
            'affected_individual': 'Affected Individual',
            'caution_and_advice': 'Caution and Advice',
            'displaced_and_evacuations': 'Displaced and Evacuations',
            'donation_and_volunteering': 'Donation and Volunteering',
            'infrastructure_and_utilities_damage': 'Infrastructure and Utilities Damage',
            'injured_or_dead_people': 'Injured or Dead People',
            'missing_and_found_people': 'Missing and Found People',
            'not_humanitarian': 'Not Humanitarian',
            'requests_or_needs': 'Requests or Needs',
            'response_efforts': 'Response Efforts',
            'sympathy_and_support': 'Sympathy and Support',
            'displaced_people_and_evacuations': 'Displaced and Evacuations',
            'infrastructure_and_utility_damage': 'Infrastructure and Utilities Damage',
            'missing_or_found_people': 'Missing and Found People',
            'requests_or_urgent_needs': 'Requests or Needs',
            'rescue_volunteering_or_donation_effort': 'Donation and Volunteering',
            'other_relevant_information': 'Caution and Advice',
        }
        self.inverse_label_mapping = {}
        for k, v in self.label_mapping.items():
            if v not in self.inverse_label_mapping:
                self.inverse_label_mapping[v] = k
        self.in_context_examples, self.in_context_indexes, self.in_context_model,  = self.load_in_context_examples()


    def get_in_context_examples(self, posts: List[dict], scope: str = "all", num: int = 5):
        texts = [post["text"] for post in posts]
        embeddings = self.in_context_model.encode(texts, show_progress_bar=False)
        # Find 3 most similar sentences of the corpus for each query sentence based on cosine similarity
        D, I = self.in_context_indexes[scope].index.search(embeddings, num)

        contexts = []
        for idx in range(len(texts)):
            contexts.append([self.in_context_examples[self.in_context_indexes[scope].mapping[i]] for i in I[idx]])
        return contexts

    def get_in_context_examples_per_class(self, posts: List[dict], num=1):
        all_contexts = []
        to_ignore = {"all"}
        if self.ignore_negative:
            to_ignore.add("Not Humanitarian")
        for class_label in (self.in_context_indexes.keys() - to_ignore):
            contexts = self.get_in_context_examples(posts, scope=class_label, num=num)
            all_contexts.append(contexts)
        combined_contexts = []
        for elem in zip(*all_contexts):
            context = list(itertools.chain(*elem))
            combined_contexts.append(context)
        return combined_contexts



    def load_in_context_examples(self):
        paths = ["data/event_data/cbd/translated_train.json",
                 "data/event_data/cbd/translated_dev.json",
                 "data/event_data/cbd/translated_test.json",
                 "data/event_data/HumAID/translated_train.json",
                 "data/event_data/HumAID/translated_dev.json",
                 "data/event_data/HumAID/translated_test.json", ]

        all_examples = {}
        sentences = []
        per_class = defaultdict(list)
        counter = 0
        for path in paths:
            data = json.load(open(path))
            for elem in data:
                class_label = self.label_mapping[elem["class_label"]]
                all_examples[counter] = {
                    "text": elem["german_translation"],
                    "class": class_label}
                sentences.append(elem["german_translation"])
                per_class[class_label].append(counter)
                counter += 1
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(sentences, show_progress_bar=True)

        # Populate a efficient faiss index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        indexes = {"all": IndexTuple(index, {idx: idx for idx in range(len(sentences))})}

        for class_label, indices in per_class.items():
            indices = np.array(indices)
            embeddings_class = embeddings[indices]
            index_class = faiss.IndexFlatL2(embeddings_class.shape[1])
            index_class.add(embeddings_class)
            indexes[class_label] = IndexTuple(index_class, {idx: indices[idx] for idx in range(len(indices))})

        return all_examples, indexes, model

    def parse_output(self, response: str):
        categories = []
        for line in response.split("\n"):
            if "Categories:" in line:
                for cat in self.inverse_label_mapping.keys():
                    if cat in line:
                        categories.append(self.inverse_label_mapping[cat])

        categories = list(set(categories))
        return categories


    def predict(self, posts: List[dict]) -> List[Dict]:
        if self.balanced_few_shots:
            in_context_examples = self.get_in_context_examples_per_class(posts, num=self.num_examples_per_class)
        else:
            in_context_examples = self.get_in_context_examples(posts, scope="all", num=self.num_examples_per_class)

        messages = []
        for post, context in tqdm(zip(posts, in_context_examples)):
            if post.get("images", []):
                complete_prompt = self.mm_prompt + create_examples(context) + "If you are uncertain about a post's classification, review the definitions carefully and assign the most appropriate categories."
                text = post["text"]
            else:
                complete_prompt = self.prompt + create_examples(context) + "If you are uncertain about a post's classification, review the definitions carefully and assign the most appropriate categories."
                text = post["text"]

            images = post.get("images", [])
            user_turn = {"role": "user", "content": [{"type": "text", "text": "Input: " + text}]}
            if images:
                for image in images[:3]:
                    image = base64.b64encode(image).decode("utf-8")
                    user_turn["content"].append({"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }})


            prompt = [{"role": "system", "content": complete_prompt},
                      user_turn]

            messages.append(prompt)

        responses = self.llm.generate(messages=messages)
        categorized = []
        for response in responses:
            categorized.append({
                "predicted_labels": self.parse_output(response),
            })
        return categorized

    def generate_annotations(self, file_path: str, output_path: str):
        social_media_posts = json.load(open(file_path))

        texts = []
        for item in social_media_posts:
            texts.append(item["text"])

        output = self.predict(texts)

        categorized = []
        for post, x in zip(social_media_posts, output):
            categorized.append({"post": post,
                                **x})

        json.dump(categorized, open(output_path), indent=2)


def create_examples(context: list):
    random.shuffle(context)
    prompt = "\n\n"
    prompt += "**Examples:**\n"
    for idx, elem in enumerate(context):
        prompt += f"{1+idx}.\n```\nInput: \"{elem['text']}\"\nExplanation: ...\nCategories: {elem['class']}\n```\n\n"

    return prompt







