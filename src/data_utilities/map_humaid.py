import argparse
import json
import os

import pandas as pd


def main(path: str):
    train_path = path + "/all_combined/all_train.tsv"
    dev_path = path + "/all_combined/all_dev.tsv"
    test_path = path + "/all_combined/all_test.tsv"

    train_ids = set(pd.read_csv(train_path, sep="\t")["tweet_id"].values)
    dev_ids = set(pd.read_csv(dev_path, sep="\t")["tweet_id"].values)
    test_ids = set(pd.read_csv(test_path, sep="\t")["tweet_id"].values)

    all_tweets = []
    for data_directory in ["/events_set1", "/events_set2"]:
        for folder in os.listdir(path + data_directory):
            if not os.path.isdir(path + data_directory + "/" + folder):
                continue
            for file in os.listdir(path + data_directory + "/" + folder):
                if file.endswith(".tsv"):
                    tweets = pd.read_csv(path + data_directory + "/" + folder + "/" + file, sep="\t")
                    all_tweets.append(tweets)
    all_tweets = pd.concat(all_tweets)

    train_set = []
    dev_set = []
    test_set = []
    for tweet_id, tweet_text, tweet_label in zip(all_tweets["tweet_id"], all_tweets["tweet_text"], all_tweets["class_label"]):
        if tweet_id in train_ids:
            train_set.append({"text": tweet_text, "class_label": tweet_label, "tweet_id": tweet_id})
        elif tweet_id in dev_ids:
            dev_set.append({"text": tweet_text, "class_label": tweet_label, "tweet_id": tweet_id})
        elif tweet_id in test_ids:
            test_set.append({"text": tweet_text, "class_label": tweet_label, "tweet_id": tweet_id})

    json.dump(train_set, open(path + "/train.json", "w"), indent=2)
    json.dump(dev_set, open(path + "/dev.json", "w"), indent=2)
    json.dump(test_set, open(path + "/test.json", "w"), indent=2)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", type=str, default="data/event_data/HumAID")
    args = argparser.parse_args()
    main(args.path)