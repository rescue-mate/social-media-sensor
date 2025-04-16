import argparse
import json
from csv import DictReader


def main(path: str):
    train = []
    for row in DictReader(open(path + "/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_train.tsv"), delimiter="\t"):
        train.append(row)

    dev = []
    for row in DictReader(open(path + "/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_dev.tsv"), delimiter="\t"):
        dev.append(row)

    test = []
    for row in DictReader(open(path + "/all_data_en/crisis_consolidated_humanitarian_filtered_lang_en_test.tsv"), delimiter="\t"):
        test.append(row)

    json.dump(train, open(path + "/train.json", "w"), indent=2)
    json.dump(dev, open(path + "/dev.json", "w"), indent=2)
    json.dump(test, open(path + "/test.json", "w"), indent=2)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path", type=str, default="data/event_data/cbd")
    args = argparser.parse_args()
    main(args.path)