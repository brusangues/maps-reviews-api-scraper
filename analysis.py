import pandas as pd
from pathlib import Path
import json
import os

from src.analysis_config import *

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3000"
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

data_path = Path("data/2022/11/20/")
places_file = "data/places.csv"
input_file = "input/hotels.csv"


def main():

    # Reading input
    input = pd.read_csv(input_file, sep=",", encoding="utf-8")
    input = input[input_cols]
    input = input.rename(columns={"name": "name_input"})

    # Reading metadata
    metadata = []
    for f in data_path.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["file_name"] = f.stem
        metadata.append(d)

    metadata = pd.DataFrame.from_records(metadata)
    metadata = metadata.rename(columns={"retrieval_date": "retrieval_date_metadata"})
    print(metadata)
    metadata.info()

    # Reading data
    dfs = []
    for f in data_path.glob("*.csv"):
        df = pd.read_csv(f, sep=",", encoding="utf-8")
        df["file_name"] = f.stem
        dfs.append(df)

    reviews = pd.concat(dfs, axis=0)
    print(reviews)
    reviews.info()

    # Merging
    df = input.merge(metadata, on="url", how="left", validate="one_to_many")
    df = df[~df.file_name.isna()]
    df = df.merge(reviews, on="file_name", validate="one_to_many")

    print(df)
    df.info()

    ### Some stats
    # Checking for errors
    df.errors.value_counts()

    # Checking for duplicate reviews
    duplicate_ids = df[df.review_id.duplicated()].review_id
    df[df.review_id.isin(duplicate_ids)]

    # Dropping duplicate
    df = df.drop_duplicates(subset="review_id")

    # Counting actual number of reviews
    df_count = df.groupby(["name"]).agg(agg_dict)
    df_count[["review_id", "n_reviews"]]
    df_count[df_count.review_id < df_count.n_reviews][["review_id", "n_reviews"]]
    df_count.to_excel("analysis.xlsx", writer="openpyxl")


if __name__ == "__main__":
    main()
