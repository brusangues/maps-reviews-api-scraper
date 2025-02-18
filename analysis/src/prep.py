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
from tqdm import tqdm

from analysis.src.config import (
    input_cols,
    agg_dict,
    text_agg_dict,
    feature_cols,
    text_cols,
    full_cols,
    relative_date_maps,
    translated_text_maps,
    csv_reviews_cols,
)
from analysis.src.preprocessing import map_progress, tokenizer_lemma

tqdm.pandas()


def read_data(input_file: str, data_path: str) -> pd.DataFrame:
    print("read_data...")

    print("Lendo arquivo de input")
    input = pd.read_csv(input_file, sep=",", encoding="utf-8")
    print(f"{input.shape=}")
    # input = input[input_cols]
    input = input.rename(columns={"name": "name_input", "n_reviews": "n_reviews_input"})
    print(f"{input.columns=}")

    print("Lendo metadados dos hotéis baixados no início da raspagem")
    metadata = []
    jsons = list(Path(data_path).glob("*.json"))
    for f in tqdm(jsons, total=len(jsons)):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["file_name"] = f.stem
        metadata.append(d)

    metadata = pd.DataFrame.from_records(metadata)
    print(f"{metadata.shape=}")
    metadata = metadata.rename(columns={"retrieval_date": "retrieval_date_metadata"})
    print(f"{metadata.columns=}")

    print("Lendo avaliações dos hotéis")
    dfs = []
    csvs = list(Path(data_path).glob("*.csv"))
    for f in tqdm(csvs, total=len(csvs)):
        df = pd.read_csv(
            f,
            sep=",",
            encoding="utf-8",
            names=csv_reviews_cols,
            header=None,
            low_memory=False,
            skiprows=0,
        )
        df["file_name"] = f.stem
        dfs.append(df)
    df_reviews = pd.concat(dfs, axis=0)
    print(f"{df_reviews.shape=}")
    print(f"{df_reviews.columns=}")

    print("Removendo lixo das avaliações dos hotéis")
    df_reviews = df_reviews.loc[~(df_reviews.iloc[:, 0] == csv_reviews_cols[0])]
    print(f"{df_reviews.shape=}")

    print("Unindo as diferentes fontes de dados")
    print("Merge input metadata")
    df_hotels = input.merge(metadata, on="url", how="left", validate="one_to_many")
    print(f"{df_hotels.shape=}")
    print("Removendo hotéis sem metadados (file_name.isna())")
    df_hotels = df_hotels[~df_hotels.file_name.isna()]
    print(f"{df_hotels.shape=}")
    print("Merge avaliações")
    df_merge = df_hotels.merge(df_reviews, on="file_name", validate="one_to_many")
    print(f"{df_merge.shape=}")
    print(f"{df_merge.columns=}")

    return df_merge


def prep_complete(
    df_merge: pd.DataFrame, lat_long_path: str = "analysis/artifacts/lat_long.csv"
):
    df = df_merge.copy()
    print(f"{df.shape=}")

    print("drop duplicates")
    df = df.drop_duplicates(subset="review_id").reset_index(drop=True)
    print(f"{df.shape=}")

    print("likes")
    df.loc[:, "likes"] = df.likes.replace({-1: 0}).fillna(0)
    print(f"{df.likes.value_counts()=}")

    print("trip_type_travel_group")  # trip_type_travel_group
    df.loc[:, "trip_type_travel_group"] = df.trip_type_travel_group.apply(
        lambda x: x if pd.isna(x) else x.split(" | ")
    )
    print(f"{df.trip_type_travel_group.value_counts()=}")

    # print("lattitude and longitude")
    # lat_long = pd.read_csv(lat_long_path)
    # df = df.merge(lat_long, on="name", how="inner", validate="many_to_one")

    print("relative dates")
    df.loc[:, "review_date"] = df.progress_apply(
        lambda x: parse_relative_date(x.relative_date, x.retrieval_date),
        axis=1,
    )
    df.loc[:, "response_date"] = df.progress_apply(
        lambda x: parse_relative_date(x.response_relative_date, x.retrieval_date),
        axis=1,
    )

    print("Other ratings")
    df[["rooms", "locale", "service", "highlights"]] = df.progress_apply(
        lambda x: parse_other_ratings(x["other_ratings"]),
        axis=1,
        result_type="expand",
    )

    # print("Topics")
    # df[["topic_names", "topic_counts"]] = df.progress_apply(
    #     lambda x: parse_topics(x.topics),
    #     axis=1,
    #     result_type="expand",
    # )

    print("User")
    df["user_id"] = df.user_url.progress_apply(parse_user_id)

    # print("Text")
    # df[["text_is_other_language", "text"]] = df.progress_apply(
    #     lambda x: parse_translated_text(x["text"]),
    #     axis=1,
    #     result_type="expand",
    # )
    # df[["response_text_is_other_language", "response_text"]] = df.progress_apply(
    #     lambda x: parse_translated_text(x["response_text"]),
    #     axis=1,
    #     result_type="expand",
    # )

    # print("tokens")
    # df.loc[:, "text_tokens"] = map_progress(tokenizer_lemma, df.text)
    # df.loc[:, "text_tokens_len"] = df["text_tokens"].apply(len)
    # df.loc[:, "response_text_tokens"] = map_progress(tokenizer_lemma, df.response_text)
    # df.loc[:, "response_text_tokens_len"] = df["response_text_tokens"].apply(len)

    return df


# User
def parse_user_id(user_url):
    if pd.isna(user_url):
        return None
    user_id = re.search("(?<=[/])\d{21}(?=[?])", user_url)
    if user_id:
        return user_id[0]
    return None


# Topics
def parse_topics(text):
    text = re.sub("Todas 0 |[+]6 ", "", text)
    text = re.sub("2016", "'2016", text)
    topics = re.split("(?<= \d+) ", text)
    topics = [re.split(" (?=\d+)", t) for t in topics]
    topic_names = [x[0] for x in topics]
    topic_counts = [int(x[1]) for x in topics]
    return topic_names, topic_counts


# Other ratings
def parse_category(regex, text):
    score = re.search(regex, text)
    if score:
        return float(score[0])
    return None


def parse_other_ratings(text):
    if not isinstance(text, str):
        return None, None, None, None
    locale = parse_category("(?<=Local : )\d(?=/)", text)
    rooms = parse_category("(?<=Quartos : )\d(?=/)", text)
    service = parse_category("(?<=Serviço : )\d(?=/)", text)

    highlights = re.search("(?<=Destaques do hotel ).*", text)
    if highlights:
        highlights = re.split(", | e ", highlights[0])
    else:
        highlights = None
    return rooms, locale, service, highlights


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
        return None, None
    is_other_language = translated_text_maps[hl]["flag"] in text
    if is_other_language:
        text = re.sub(translated_text_maps[hl]["regex"], "", text)
        text = text.strip()
    return is_other_language, text


features = [
    # Hotel Features
    "feature_id",
    "name",
    "retrieval_date_metadata",
    "state",
    "region",
    "latitude",
    "longitude",
    "stars",
    "overall_rating",
    "n_reviews",
    "topic_names",
    "topic_counts",
    # Review Structured
    "review_id",
    "retrieval_date",
    "review_date",
    "response_date",
    "rating",
    "likes",
    "rooms",
    "locale",
    "service",
    # User Features
    "user_id",
    "user_name",
    "user_is_local_guide",
    "user_reviews",
    "user_photos",
    # Word lists
    "highlights",
    "trip_type_travel_group",
    # Text features
    "text",
    "response_text",
    "text_is_other_language",
    "response_text_is_other_language",
    "text_tokens",
    "text_tokens_len",
    "response_text_tokens",
    "response_text_tokens_len",
]
