import pandas as pd
import csv
import os

def prepare_dataset(input_path, output_path):
    future_df = {"filepath":[], "title":[]}
    for row in csv.DictReader(open(input_path, "r", encoding="utf-8"), delimiter="\t"):
        future_df["filepath"].append(row["image_path"])
        future_df["title"].append(row["class_label"])
    pd.DataFrame.from_dict(future_df).to_csv(
      output_path, index=False, sep="\t"
    )


if __name__ == "__main__":
    prepare_dataset("data_humanitarian/consolidated_hum_train_final.tsv","data_humanitarian/train.csv")
    prepare_dataset("data_humanitarian/consolidated_hum_dev_final.tsv","data_humanitarian/dev.csv")