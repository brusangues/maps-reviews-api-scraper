import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import regex as re
import typer
from dateutils import relativedelta
from unidecode import unidecode

from analysis.config import *
from analysis.preprocessing import map_progress, tokenizer_lemma
from analysis.utils import *
from src.custom_logger import get_logger

app = typer.Typer()
logger = logging.getLogger()
logger = get_logger("analysis", logger=logger)

# Removendo aviso de debug
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3000"
# Removendo limitação de print de dfs
# pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
# pd.set_option("display.width", 1000)

# Caminho da pasta contendo os csvs
data_path = "data/2023/01/19/"
places_file = "data/places.csv"
input_file = "input/hotels_23_01_19.csv"
reports_folder = Path("./reports")
Path(reports_folder).mkdir(exist_ok=True)
data_folder = Path("./data/prep")
Path(data_folder).mkdir(exist_ok=True)


@app.command()
def main(input_file=input_file, data_path=data_path):
    df = read_data(input_file, data_path)

    ### Some stats
    # Checking for errors
    df.errors.value_counts()

    # Checking for duplicate reviews
    duplicate_ids = df[df.review_id.duplicated()].review_id
    df_duplicates = df[df.review_id.isin(duplicate_ids)]
    make_report(df_duplicates, reports_folder, "df_duplicates")

    # Dropping duplicate
    df = df.drop_duplicates(subset="review_id")
    save_df(df, data_folder, "df_raw")

    # Counting actual number of reviews
    df_count = df.groupby(["name"]).agg(agg_dict)
    df_count[["review_id", "n_reviews"]]
    df_count[df_count.review_id < df_count.n_reviews][["review_id", "n_reviews"]]
    df_count["n_reviews_diff"] = df_count.n_reviews - df_count.review_id
    make_report(df_count, reports_folder, "df_count")

    df_count["text_percentage"] = df_count["text"] / df_count["review_id"]
    df_count[["text_percentage", "text", "review_id"]]
    df_count2 = df_count.agg(
        {"text_percentage": "mean", "text": "sum", "review_id": "sum"}
    )

    # # ### Text processing
    # df_text = df[text_cols]

    # # Dates
    # df_text["review_date"] = df_text.apply(
    #     lambda x: parse_relative_date(x.relative_date, x.retrieval_date),
    #     axis=1,
    #     # result_type="expand",
    # )
    # df_text[["retrieval_date", "relative_date", "review_date"]]
    # df_text["response_date"] = df_text.apply(
    #     lambda x: parse_relative_date(x.response_relative_date, x.retrieval_date),
    #     axis=1,
    #     # result_type="expand",
    # )
    # df_text[["retrieval_date", "response_relative_date", "response_date"]]

    # make_report(df_text, reports_folder, "df_text")

    # # Language
    # # "(Tradução do Google) Topo (Original) Top"
    # # (Translated by Google) Nice (Original) Agradável
    # # df_text["is_other_language"] = df_text.text.str.contains("Tradução do Google")
    # # df_text["is_other_language"] = df_text["is_other_language"].fillna(False)
    # df_text[["is_other_language", "text"]] = df_text.apply(
    #     lambda x: parse_translated_text(x.text),
    #     axis=1,
    #     result_type="expand",
    # )
    # df_text["is_other_language"].sum()
    # df_text[df_text["is_other_language"]].text

    # # Tokens
    # df_text["tokens"] = map_progress(tokenizer_lemma, df_text.text)
    # df_text["tokens_len"] = df_text["tokens"].apply(len)
    # df_text[["text", "tokens", "tokens_len"]]

    # make_report(df_text, "df_text")

    # Features
    # df_features = df[feature_cols]


if __name__ == "__main__":
    app()
