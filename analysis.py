import pandas as pd
from pathlib import Path
import json
import os
from datetime import datetime
import regex as re
from unidecode import unidecode
from dateutils import relativedelta

from src.analysis_config import *
from src.analysis_preprocessing import map_progress, tokenizer_lemma

# Removendo aviso de debug
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "3000"
# Removendo limitação de print de dfs
# pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
# pd.set_option("display.width", 1000)

# Caminho da pasta contendo os csvs
data_path = Path("data/2022/11/23/")
places_file = "data/places.csv"
input_file = "input/hotels.csv"
reports_folder = "./reports"
Path(reports_folder).mkdir(exist_ok=True)

# Time stamp
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def make_report(df, name):
    """Escreve arquivo excel consolidado"""
    if df.empty:
        return
    file_path = f"{reports_folder}/{name}_{ts}.xlsx"
    return df.to_excel(file_path, engine="openpyxl")


def parse_relative_date(relative_date, retrieval_date, hl="pt-br"):
    """Transforma data relativa do google maps em datetime"""
    if (not isinstance(relative_date, str)) or relative_date == "":
        return pd.NaT
    # Normaliza texto
    text = unidecode(relative_date).lower().strip()
    # Transforma {"um","uma"} no número 1
    text = re.sub(relative_date_maps[hl]["one_regex"], "1", text)
    # Remove terminação "atrás"
    text = re.sub(relative_date_maps[hl]["ago_regex"], "", text)

    number, time_unit = text.split(" ")
    number = float(number)
    kwargs = {relative_date_maps[hl]["time_unit"][time_unit]: number}
    review_date = pd.to_datetime(retrieval_date) - relativedelta(**kwargs)
    return review_date


def parse_translated_text(text, hl="pt-br"):
    if not isinstance(text, str):
        return False, None
    is_other_language = translated_text_maps[hl]["flag"] in text
    if is_other_language:
        text = re.sub(translated_text_maps[hl]["regex"], "", text)
        text = text.strip()
    return is_other_language, text


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
    df_duplicates = df[df.review_id.isin(duplicate_ids)]
    make_report(df_duplicates, "df_duplicates")

    # Dropping duplicate
    df = df.drop_duplicates(subset="review_id")

    # Counting actual number of reviews
    df_count = df.groupby(["name"]).agg(agg_dict)
    df_count[["review_id", "n_reviews"]]
    df_count[df_count.review_id < df_count.n_reviews][["review_id", "n_reviews"]]
    df_count["n_reviews_diff"] = df_count.n_reviews - df_count.review_id
    make_report(df_count, "df_count")

    # ### Text processing
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
    main()
