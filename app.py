from datetime import datetime
from pathlib import Path
import csv
import json
from multiprocessing import Pool
import pandas as pd
import typer

from src.scraper import GoogleMapsAPIScraper
from src.config import review_default_result, metadata_default
from src.custom_logger import get_logger

app = typer.Typer()
logger = get_logger("google_maps_api_scraper")

n_processes = 8
file_path = "input/hotels.csv"
places_path = "data/places.csv"


@app.command()
def run(path: str = file_path):
    df_list = load_input(path)
    results = call_sequential(df_list)
    log_summary(results, df_list)


@app.command()
def run_async(path: str = file_path):
    df_list = load_input(path)
    results = call_pools(df_list)
    log_summary(results, df_list)


def call_sequential(df_list: list) -> list:
    logger.info("Running sync")
    results = []
    for row in df_list:
        result = call_scraper(**row)
        results.append(result)
    logger.info("Finished running scraper sync")
    return results


def call_pools(df_list: list) -> list:
    logger.info("Running async")
    results = []
    with Pool(processes=n_processes) as pool:
        for row in df_list:
            p = pool.apply_async(
                func=call_scraper,
                kwds=row,
            )
            results.append(p)
        [p.wait() for p in results]
        results = [p.get() for p in results]
    logger.info("Finished running scraper async")
    return results


def load_input(path: str):
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    df = df.loc[df.done == 0]
    df_list = df.to_dict(orient="records")
    return df_list


def log_summary(results: list, df_list: list):
    for ((rs, m), row) in zip(results, df_list):
        logger.info(
            f"name:{m['name']:<16.16}; "
            f"place_name:{m['place_name']:<16.16}; "
            f"n_max:{m['n_reviews']:>6}; "
            f"n_input:{row['n_reviews']:>6}; "
            f"n_scraped:{len(rs):>6}; "
            f"n_errors:{len([e for r in rs for es in r['errors'] for e in es])}"
        )


def call_scraper(name: str, n_reviews: int, url: str, sort_by: str, hl: str, **kwargs):
    # Create date folder
    path = datetime.now().strftime("data/%Y/%m/%d/")
    Path(path).mkdir(exist_ok=True, parents=True)
    logger.info("folder created")

    # Make filename
    file_name = str(name).strip().lower().replace(" ", "-")
    reviews_file_name = file_name + "-gm-reviews.csv"
    place_file_name = file_name + "-gm-reviews.json"

    # Clear file contents
    # with open(path + reviews_file_name, "w") as f:
    #     pass

    # Create scraper object
    with GoogleMapsAPIScraper(hl=hl, logger=logger) as scraper:
        # Create csv writer for metadata
        write_places_header = not Path(places_path).exists()
        with open(places_path, "a+", encoding="utf-8", newline="\n") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            if write_places_header:
                writer.writerow(metadata_default.keys())

            metadata = scraper.scrape_place(url, writer, file, name)

        # Create json for metadata
        with open(path + place_file_name, "w", encoding="latin1") as f:
            json.dump(metadata, f, indent=4)

        # Changes negative n_reviews
        if n_reviews < 0:
            n_reviews = metadata["n_reviews"]

        # Create csv writer and start scraping
        with open(
            path + reviews_file_name, "a+", encoding="utf-8", newline="\n"
        ) as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(review_default_result.keys())
            logger.info("header written")

            try:
                reviews = scraper.scrape_reviews(
                    url, writer, file, n_reviews, sort_by=sort_by
                )
            except Exception as e:
                logger.exception("Error in scraper.scrape_reviews")
                raise

    return reviews, metadata


if __name__ == "__main__":
    app()
