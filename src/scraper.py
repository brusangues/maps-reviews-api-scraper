from typing import List, Tuple, Union
import requests
import traceback
from datetime import datetime
import time
import math
import urllib.parse
from pathlib import Path
from lxml import etree, html
from bs4 import BeautifulSoup, Tag, NavigableString, Comment
import regex as re

from src.customlogger import get_logger
from src.config import sort_by_enum, review_default_result, metadata_default

default_hl = "pt-br"
default_request_interval = 0.5
default_n_retries = 10
default_retry_time = 30

Path("examples/").mkdir(exist_ok=True)
Path("errors/").mkdir(exist_ok=True)


class GoogleMapsAPIScraper:
    def __init__(
        self,
        hl: str = default_hl,
        request_interval: float = default_request_interval,
        n_retries: int = default_n_retries,
        retry_time: float = default_retry_time,
        logger=None,
    ):
        if not logger is None:
            self.logger = logger
        else:
            self.logger = get_logger("google_maps_api_scraper")
        self.hl = hl
        self.request_interval = request_interval
        self.n_retries = n_retries
        self.retry_time = retry_time

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            self.logger.exception(exc_value)

        return True

    def _ts(self) -> str:
        """Returns timestamp formatted as string safe for file naming"""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    def _parse_url_to_feature_id(self, url: str) -> str:
        return re.findall("0[xX][0-9a-fA-F]+:0[xX][0-9a-fA-F]+", url)[0]

    def _parse_sort_by(self, sort_by: str) -> int:
        """Default to newest"""
        return sort_by_enum.get(sort_by, 1)

    def _decode_response(self, response) -> str:
        """Decodes response bytes in unicode escape encoding"""
        try:
            response_text = response.content.decode(encoding="unicode_escape")
        except UnicodeDecodeError as e:
            tb = re.sub("\s", " ", traceback.format_exc())
            self.logger.info(f"UnicodeDecodeError. Replacing errors. {tb}")
            response_text = response.content.decode(
                encoding="unicode_escape", errors="replace"
            )
        if response_text is None or response_text == "":
            raise Exception(
                "Response text is none. Try request again."
                f"Response: {response} Status: {response.status_code}"
            )
        return response_text

    def _cut_response_text(self, text: str) -> str:
        """Cut response text to remove css and js from extremities"""
        idx_first_div = text.find("<div")
        if idx_first_div == -1:
            self.logger.info("first div not found")
            idx_first_div = 0
        match = re.search("</div", text, flags=re.REVERSE)
        if match:
            idx_last_div = match.span()[1] + 1
        else:
            self.logger.info("last div not found")
            idx_last_div = -1
        text = text[idx_first_div:idx_last_div]
        return "<html><body>" + text + "</body></html>"

    def _format_response_text(self, response_text: str):
        """Transforms text into soup and extract list of reviews"""
        response_soup = reviews_soup = review_count = next_token = None
        try:
            # Send page to soup and trees
            response_soup = BeautifulSoup(response_text, "lxml")
            tree = html.document_fromstring(response_text)

            # Encontrando número de reviews e token de próxima página
            metadata_node = tree.xpath("//*[@data-google-review-count]")[0]
            review_count = int(metadata_node.attrib["data-google-review-count"])
            next_token = metadata_node.attrib["data-next-page-token"]

            # Iterando sobre texto de cada review
            reviews_tree = tree.xpath(
                "/html/body/div[1]/div/div[2]/div[4]/div/div[2]/div"
            )
            reviews_soup = [
                response_soup.find("div", dict(r.attrib)) for r in reviews_tree
            ]
        except Exception as e:
            tb = re.sub("\s", " ", traceback.format_exc())
            self.logger.info(f"Response formatting error: {tb}")
            if next_token is None:
                next_token = self._get_response_token(response_text)

        return response_text, response_soup, reviews_soup, review_count, next_token

    def _get_response_token(self, response_text: str) -> str:
        """Searches for token in response text using regex, in case other methods fail"""
        match = re.search('(data-next-page-token\s*=\s*")([\w=]*)', response_text)
        if match:
            return match.groups()[1]
        self.logger.info("regex token not found")

    def _get_request(
        self,
        feature_id: str,
        async_: str = "",
        hl: str = "",
        sort_by_id: int = "",
        associated_topic: str = "",
        token: str = "",
    ) -> Tuple[BeautifulSoup, List[BeautifulSoup], int, str]:
        """Makes and formats get request in google's api"""
        if not hl:
            hl = self.hl
        query = (
            "https://www.google.com/async/reviewDialog?"
            f"hl={hl}&"
            f"async={async_}"
            f"feature_id:{feature_id},"
            f"sort_by:{sort_by_id},"
            f"next_page_token:{token},"
            f"associated_topic:{associated_topic},"
            f"_fmt:pc"
        )
        # Make request
        response = requests.get(query)
        response.raise_for_status()

        # Decode response
        response_text = self._decode_response(response)

        # Cut response to remove css
        response_text = self._cut_response_text(response_text)

        # Format response into list of reviews
        return self._format_response_text(response_text)

    def _parse_place(
        self,
        response: BeautifulSoup,
    ) -> dict:
        """Parse place html"""
        metadata = metadata_default.copy()

        # Parse place_name
        try:
            metadata["place_name"] = response.find(True, class_="P5Bobd").text
        except Exception as e:
            self.logger.error("error parsing place: place_name")
            self.logger.exception(e)

        # Parse address
        try:
            metadata["address"] = response.find(True, class_="T6pBCe").text
        except Exception as e:
            self.logger.error("error parsing place: address")
            self.logger.exception(e)

        # Parse overall_rating
        try:
            rating_text = response.find(True, class_="Aq14fc").text.replace(",", ".")
            metadata["overall_rating"] = float(rating_text)
        except Exception as e:
            self.logger.error("error parsing place: overall_rating")
            self.logger.exception(e)

        # Parse n_reviews
        try:
            n_reviews_text = response.find(True, class_="z5jxId").text
            n_reviews_text = re.sub("[.]|,| reviews| comentários", "", n_reviews_text)
            metadata["n_reviews"] = int(n_reviews_text)
        except Exception as e:
            self.logger.error("error parsing place: n_reviews")
            self.logger.exception(e)

        # Parse topics
        try:
            topics = response.find("localreviews-place-topics")
            s = " ".join([s for s in topics.stripped_strings])
            metadata["topics"] = re.sub("\s+", " ", s)
        except Exception as e:
            self.logger.error("error parsing place: topics")
            self.logger.exception(e)

        metadata["retrieval_date"] = str(datetime.now())

        return metadata

    def _parse_review_text(self, text_block) -> str:
        """Parse review text html, removing unwanted characters"""
        text = ""
        for e, s in zip(text_block.contents, text_block.stripped_strings):
            if isinstance(e, Tag) and e.has_attr(
                "class"
            ):  #  and e.attrs["class"] in ["review-snippet","k8MTF",]:
                break
            text += s + " "

        text = re.sub("\s", " ", text)
        text = re.sub("'|\"", "", text)
        text = text.strip()
        return text

    def _handle_review_exception(self, result, review, name) -> dict:
        # Error log
        tb = re.sub("\s", " ", traceback.format_exc())
        msg = f"review {name}: {tb}"
        self.logger.error(msg)
        # Appending to line
        tb = re.sub("['\"]", " ", tb)
        result["errors"].append(tb)
        # Saving file
        with open(
            f"errors/review_{name}_{self._ts()}.html", "w", encoding="utf-8"
        ) as f:
            f.writelines(str(review))
            f.writelines(msg)
        return result

    def _handle_place_exception(self, response_text, name, n) -> dict:
        # Error log
        tb = re.sub("\s", " ", traceback.format_exc())
        msg = f"place {name} request {n}: {tb}"
        self.logger.error(msg)
        # Saving file
        with open(
            f"errors/place_{name}_request_{n}_{self._ts()}.html", "w", encoding="utf-8"
        ) as f:
            f.writelines(response_text)
            f.writelines(msg)

    def _parse_review(self, review: Tag) -> dict:
        result = review_default_result.copy()

        # Make timestamp
        result["retrieval_date"] = str(datetime.now())

        # Parse text
        try:
            # Find text block
            text_block = review.find(True, class_="review-full-text")
            if not text_block:
                text_block = review.find(True, {"data-expandable-section": True})
            # Extract text
            if text_block:
                result["text"] = self._parse_review_text(text_block)
        except Exception as e:
            self._handle_review_exception(result, review, "text")

        # Parse review rating
        try:
            rating_text = review.find(True, class_="Fam1ne EBe2gf").get("aria-label")
            rating_text = re.sub(",", ".", rating_text)
            rating = re.findall("[0-9]+[.][0-9]*", rating_text)
            result["rating"] = float(rating[0])
            result["rating_max"] = float(rating[1])
        except Exception as e:
            self._handle_review_exception(result, review, "rating")

        # Parse other ratings
        try:
            other_ratings = review.find(True, class_="k8MTF")
            if other_ratings:
                s = " ".join([s for s in other_ratings.stripped_strings])
                result["other_ratings"] = re.sub("\s+", " ", s)
        except Exception as e:
            self._handle_review_exception(result, review, "other_ratings")

        # Parse relative date
        try:
            result["relative_date"] = review.find(True, class_="dehysf lTi8oc").text
        except Exception as e:
            self._handle_review_exception(result, review, "relative_date")

        # Parse user name
        try:
            result["user_name"] = review.find(True, class_="TSUbDb").text
        except Exception as e:
            self._handle_review_exception(result, review, "user_name")

        # Parse user metadata
        try:
            user_node = review.find(True, class_="Msppse")
            if user_node:
                result["user_url"] = user_node.get("href")
                result["user_is_local_guide"] = (
                    True if user_node.find(True, class_="QV3IV") else False
                )
                user_reviews = re.findall(
                    "[Uuma0-9.,]+(?= comentário| review)", user_node.text
                )
                user_photos = re.findall("[Uuma0-9.,]+(?= foto| photo)", user_node.text)
                if len(user_reviews) > 0:
                    result["user_reviews"] = user_reviews[0]
                if len(user_photos) > 0:
                    result["user_photos"] = user_photos[0]
        except Exception as e:
            self._handle_review_exception(result, review, "user_data")

        # Parse review id
        try:
            # result["review_id"] = review.find(True, {"data-ri": True}).get("data-ri")
            review_id = review.find(True, class_="RvU3D").get("href")
            result["review_id"] = re.findall("(?<=postId=).*?(?=&)", review_id)[0]
        except Exception as e:
            self._handle_review_exception(result, review, "review_id")

        # Parse review likes
        try:
            review_likes = review.find(True, jsname="CMh1ye")
            if review_likes:
                result["likes"] = int(review_likes.text)
        except Exception as e:
            self._handle_review_exception(result, review, "likes")

        # Parse review response
        try:
            response = review.find(True, class_="d6SCIc")
            if response:
                result["response_text"] = self._parse_review_text(response)
            response_date = review.find(True, class_="pi8uOe")
            if response_date:
                result["response_relative_date"] = response_date.text
        except Exception as e:
            self._handle_review_exception(result, review, "response")

        # Parse trip_type_travel_group
        try:
            trip_type_travel_group = review.find(True, class_="PV7e7")
            if trip_type_travel_group:
                s = " ".join([s for s in trip_type_travel_group.stripped_strings])
                result["trip_type_travel_group"] = re.sub("\s+", " ", s)
        except Exception as e:
            self._handle_review_exception(result, review, "trip_type_travel_group")

        return result

    def scrape_reviews(
        self,
        url: str,
        writer,
        file,
        n_reviews: int,
        hl: str = "",
        sort_by: str = "",
        token: str = "",
    ):
        """Scrape specified amount of reviews of a place, appending results in csv"""
        url_name = re.findall("(?<=place/).*?(?=/)", url)[0]
        url_name = urllib.parse.unquote_plus(url_name)
        self.logger.info(f"Scraping url: {url_name}")

        feature_id = self._parse_url_to_feature_id(url)
        sort_by_id = self._parse_sort_by(sort_by)

        results = []
        j = 0

        n_requests = math.ceil((n_reviews) / 10)
        for i in range(n_requests):
            self.logger.info(f"{url_name}; Request: {i:>8}; review: {j:>8}")
            n = self.n_retries
            while n > 0:
                next_token = None
                try:
                    (
                        response_text,
                        response_soup,
                        reviews_soup,
                        review_count,
                        next_token,
                    ) = self._get_request(
                        feature_id,
                        hl=hl,
                        sort_by_id=sort_by_id,
                        token=token,
                    )
                    assert isinstance(reviews_soup, list)
                    break
                except Exception as e:
                    n -= 1
                    self._handle_place_exception(response_text, url_name, i)
                    if n == 0 and next_token is None:
                        self.logger.exception(
                            f"Max retries exceeded. Ending: {url_name}"
                            f"\nRequests made: {i+1}\nReviews parsed: {j}"
                        )
                        raise e
                    elif n == 0:
                        self.logger.exception(
                            f"Max retries exceeded. Skipping token: {token} for hotel: {url_name}"
                            f"\nRequests made: {i+1}\nReviews parsed: {j}"
                        )
                        break
                    else:
                        self.logger.info(f"waiting {self.retry_time} seconds")
                        time.sleep(self.retry_time)
            token = next_token
            if n == 0:
                continue

            try:
                for review in reviews_soup:
                    # self.logger.info(f"Parsing review: {j:>8}")
                    result = self._parse_review(review)
                    result["token"] = token

                    writer.writerow(result.values())
                    file.flush()

                    results.append(result)
                    # self.logger.info(results)
                    j += 1
            except Exception as e:
                tb = re.sub("\s", " ", traceback.format_exc())
                self.logger.info(f"error parsing review: {j} request: {i} tb: {tb}")

            if review_count < 10 or token == "":
                self.logger.info(f"Place review limit at {j} reviews")
                break

            # Waiting so google wont block this scraper
            time.sleep(self.request_interval)

        self.logger.info(
            f"Done Scraping Reviews\nRequests made: {i+1}\nReviews parsed: {j}"
        )

        return results

    def scrape_place(
        self,
        url: str,
        writer,
        file,
        name,
        hl: str = "",
    ):
        """Scrape place metadata, writing to csv"""
        url_name = re.findall("(?<=place/).*?(?=/)", url)[0]
        url_name = urllib.parse.unquote_plus(url_name)
        self.logger.info(f"Scraping url: {url_name}")

        feature_id = self._parse_url_to_feature_id(url)

        self.logger.info(f"Parsing metadata...")
        _, response_soup, _, _, _ = self._get_request(
            feature_id,
            hl=hl,
        )
        metadata = self._parse_place(response=response_soup)
        metadata["feature_id"] = feature_id
        metadata["url"] = url
        metadata["name"] = name

        writer.writerow(metadata.values())
        file.flush()

        return metadata
