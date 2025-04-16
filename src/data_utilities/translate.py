import argparse
import json
from csv import DictReader

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


def create_prompt():
    main_prompt = "Please translate the following tweet to German. Let it sound natural. Do this as follows.\n" \
                  "Description of what was said: ...\n" \
                   "Discussion of what needs to be changed to sound natural: ...\n" \
                  "#TRANSLATION# ... #END#\n"

    return main_prompt
def main(tweets: list):
    llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", dtype=torch.bfloat16, quantization="bitsandbytes",
              load_format="bitsandbytes", max_model_len=4028)
    tokenizer = llm.llm_engine.tokenizer.tokenizer
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
    all_prompts = []
    for post in tqdm(tweets):
        prompt = f"Translate from English to German. Let it be an idiomatic translation. If it makes sense to leave some English terms in, do that. Do only provide the translation.\nEnglish: {post['text']}"

        prompt = [{"role": "user", "content": prompt}]


        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)


        all_prompts.append(prompt_token_ids)

    responses = llm.generate(prompt_token_ids=all_prompts, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]
    translated = []
    for post, response in zip(tweets, responses):
        post["german_translation"] = response
        translated.append(post)


    return translated



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str)
    args = argparser.parse_args()
    tweets_path = args.path
    filename = tweets_path.split("/")[-1].split(".")[0]
    main_path = "/".join(tweets_path.split("/")[:-1])
    tweets = json.load(open(tweets_path))
    output_text = main(tweets)
    json.dump(output_text, open(f"{main_path}/translated_{filename}.json", "w"), indent=2)

