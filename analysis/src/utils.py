import json
import logging
import os
from datetime import datetime
import time
from pathlib import Path

import pandas as pd
import regex as re
import typer
from dateutils import relativedelta
from unidecode import unidecode

from analysis.src.config import *


def make_report(df, reports_folder, name):
    """Escreve arquivo excel consolidado"""
    if df.empty:
        return
    file_path = f"{reports_folder}/{name}_{ts}.xlsx"
    return df.to_excel(file_path, engine="openpyxl")


def save_df(df, folder, name):
    """Escreve arquivo parquet"""
    if df.empty:
        return
    file_path = f"{folder}/{name}_{ts}.pq"
    print(file_path)
    return df.to_parquet(file_path, engine="pyarrow")


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


def read_data(input_file: str, data_path: str) -> pd.DataFrame:
    # Reading input
    input = pd.read_csv(input_file, sep=",", encoding="utf-8")
    input = input[input_cols]
    input = input.rename(columns={"name": "name_input"})

    # Reading metadata
    metadata = []
    for f in Path(data_path).glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["file_name"] = f.stem
        metadata.append(d)

    metadata = pd.DataFrame.from_records(metadata)
    metadata = metadata.rename(columns={"retrieval_date": "retrieval_date_metadata"})
    logging.info(metadata)
    metadata.info()

    # Reading data
    dfs = []
    for f in Path(data_path).glob("*.csv"):
        df = pd.read_csv(f, sep=",", encoding="utf-8")
        df["file_name"] = f.stem
        dfs.append(df)

    reviews = pd.concat(dfs, axis=0)
    logging.info(reviews)
    reviews.info()

    # Merging
    df = input.merge(metadata, on="url", how="left", validate="one_to_many")
    df = df[~df.file_name.isna()]
    df = df.merge(reviews, on="file_name", validate="one_to_many")

    logging.info(df)
    df.info()
    return df


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info(f"{method.__name__} took: {te - ts}")
        return result

    return timed
