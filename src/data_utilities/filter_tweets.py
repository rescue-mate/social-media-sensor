import json
from collections import defaultdict

import torch
from torch import sigmoid
from tqdm import tqdm


def main():
    target_names = ['affected_individual', 'caution_and_advice',
                    'displaced_and_evacuations', 'donation_and_volunteering',
                    'infrastructure_and_utilities_damage', 'injured_or_dead_people',
                    'missing_and_found_people', 'not_humanitarian',
                    'requests_or_needs', 'response_efforts', 'sympathy_and_support']
    data = json.load(open("predicted.json"))
    per_category = defaultdict(list)
    filtered_data = []
    for elem in tqdm(data):
        logits = sigmoid(torch.tensor(elem["logit"])).tolist()
        text = elem["text"]
        labels = []
        scores = []
        max_idx = torch.argmax(torch.tensor(logits))
        for idx, logit in enumerate(logits):
            if logit > 0.5:
                labels.append(target_names[idx])
                scores.append(float(logit))
        if "not_humanitarian" != target_names[max_idx]:
            filtered_data.append({"text": text, "predicted_labels": labels,
                                  "scores": scores,})
            for label in labels:
                if label != "not_humanitarian":
                    per_category[label].append(text)
    json.dump(filtered_data, open("filtered.json", "w"), indent=4)

if __name__ == "__main__":
    main()