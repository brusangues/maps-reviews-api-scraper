from lxml import etree, html
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from src.scraper import GoogleMapsAPIScraper
from src.config import review_default_result
import csv
import pandas as pd
import typer

app = typer.Typer()

hl = "pt-br"
n_reviews = 40
sort_by = "newest"
feature_id = "0x94ce03043613e3d9:0x72a1063f1eb9c819"
url = "https://www.google.com/maps/place/Atl%C3%A2ntico+Inn+Apart+Hotel/@-23.9689068,-46.3317906,17z/data=!4m22!1m11!3m10!1s0x94ce03046d76cff1:0x2d62f9e79fff1d72!2sAtl%C3%A2ntico+Golden+Apart+Hotel!5m4!1s2022-11-28!2i5!4m1!1i2!8m2!3d-23.9689349!4d-46.3302516!3m9!1s0x94ce03043613e3d9:0x72a1063f1eb9c819!5m4!1s2022-11-28!2i5!4m1!1i2!8m2!3d-23.9664557!4d-46.3300101"

file_path = "input/hotels.csv"


@app.command()
def main(path: str = file_path):
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    for row in df.to_dict(orient="records"):
        call_scraper(**row)


def call_scraper(name: str, n_reviews: int, url: str, sort_by: str, hl: str):
    # Cria pasta se não existir
    path = datetime.now().strftime("data/%Y/%m/%d/")
    Path(path).mkdir(exist_ok=True, parents=True)
    print("folder created")

    file_name = str(name).strip().lower().replace(" ", "-")
    file_name += "-gm-reviews.csv"

    # Reseta arquivo
    with open(path + file_name, "w") as f:
        pass

    # Cria csv writer
    with open(path + file_name, "a+", encoding="utf-8", newline="\n") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(review_default_result.keys())
        print("header written")

        scraper = GoogleMapsAPIScraper(hl=hl)
        scraper.scrape_reviews(url, writer, file, n_reviews, sort_by=sort_by)


if __name__ == "__main__":
    app()
