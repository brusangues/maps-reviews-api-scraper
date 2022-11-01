from datetime import datetime
from pathlib import Path
import csv
import pandas as pd
import typer
from multiprocessing import Pool

from src.scraper import GoogleMapsAPIScraper
from src.config import review_default_result, metadata_default
from src.customlogger import get_logger

app = typer.Typer()
logger = get_logger("google_maps_api_scraper")

# hl = "pt-br"
# n_reviews = 40
# sort_by = "newest"
# feature_id = "0x94ce03043613e3d9:0x72a1063f1eb9c819"
# url = "https://www.google.com/maps/place/Atl%C3%A2ntico+Inn+Apart+Hotel/@-23.9689068,-46.3317906,17z/data=!4m22!1m11!3m10!1s0x94ce03046d76cff1:0x2d62f9e79fff1d72!2sAtl%C3%A2ntico+Golden+Apart+Hotel!5m4!1s2022-11-28!2i5!4m1!1i2!8m2!3d-23.9689349!4d-46.3302516!3m9!1s0x94ce03043613e3d9:0x72a1063f1eb9c819!5m4!1s2022-11-28!2i5!4m1!1i2!8m2!3d-23.9664557!4d-46.3300101"

n_processes = 4
file_path = "input/hotels.csv"
places_path = "data/places.csv"


@app.command()
def run(path: str = file_path):
    print("Running async")
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
    file_name += "-gm-reviews.csv"

    # Clear file contents
    # with open(path + file_name, "w") as f:
    #     pass

    # Create scraper object
    scraper = GoogleMapsAPIScraper(hl=hl, logger=logger)

    # Create csv writer for metadata
    write_places_header = not Path(places_path).exists()
    with open("data/places.csv", "a+", encoding="utf-8", newline="\n") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        if write_places_header:
            writer.writerow(metadata_default.keys())

        scraper.scrape_place(url, writer, file)

    # Create csv writer and start scraping
    with open(path + file_name, "a+", encoding="utf-8", newline="\n") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(review_default_result.keys())
        print("header written")

        scraper.scrape_reviews(url, writer, file, n_reviews, sort_by=sort_by)


if __name__ == "__main__":
    app()
