from datetime import datetime
from pathlib import Path
import csv
import json
from multiprocessing import Pool
import pandas as pd
import typer
import traceback

from src.scraper import GoogleMapsAPIScraper
from src.config import review_default_result, metadata_default
from src.customlogger import get_logger

app = typer.Typer()
logger = get_logger("google_maps_api_scraper")

n_processes = 8
file_path = "input/hotels.csv"
places_path = "data/places.csv"


@app.command()
def run(path: str = file_path):
    print("Running sync")
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    df = df.loc[df.done == 0]
    for row in df.to_dict(orient="records"):
        call_scraper(**row)


@app.command()
def run_async(path: str = file_path):
    print("Running async")
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    df = df.loc[df.done == 0]
    results = []
    with Pool(processes=n_processes) as pool:
        for row in df.to_dict(orient="records"):
            result = pool.apply_async(
                func=call_scraper,
                kwds=row,
            )
            results.append(result)
        [result.wait() for result in results]


def call_scraper(name: str, n_reviews: int, url: str, sort_by: str, hl: str, **kwargs):
    # Create date folder
    path = datetime.now().strftime("data/%Y/%m/%d/")
    Path(path).mkdir(exist_ok=True, parents=True)
    print("folder created")

    # Make filename
    file_name = str(name).strip().lower().replace(" ", "-")
    reviews_file_name = file_name + "-gm-reviews.csv"
    place_file_name = file_name + "-gm-reviews.json"

    # Clear file contents
    # with open(path + reviews_file_name, "w") as f:
    #     pass

    # Create scraper object
    scraper = GoogleMapsAPIScraper(hl=hl, logger=logger)

    # Create csv writer for metadata
    write_places_header = not Path(places_path).exists()
    with open(places_path, "a+", encoding="utf-8", newline="\n") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        if write_places_header:
            writer.writerow(metadata_default.keys())

        metadata = scraper.scrape_place(url, writer, file)

    # Create json for metadata
    with open(path + place_file_name, "w", encoding="latin1") as f:
        json.dump(metadata, f, indent=4)

    # Changes n_reviews
    if n_reviews < 0:
        n_reviews = metadata["n_reviews"]

    # Create csv writer and start scraping
    with open(path + reviews_file_name, "a+", encoding="utf-8", newline="\n") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(review_default_result.keys())
        print("header written")

        try:
            scraper.scrape_reviews(url, writer, file, n_reviews, sort_by=sort_by)
        except Exception as e:
            print(traceback.print_exc())
            raise e


if __name__ == "__main__":
    app()
