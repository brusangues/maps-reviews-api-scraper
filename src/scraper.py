from typing import List, Tuple
import requests
from lxml import etree, html
from bs4 import BeautifulSoup, Tag, NavigableString, Comment
import regex as re
from typing import Union
import logging
import traceback
from datetime import datetime
import time
import math
import urllib.parse

from src.customlogger import get_logger
from src.config import sort_by_enum, review_default_result


class GoogleMapsAPIScraper:
    def __init__(
        self,
        hl: str = "pt-br",
        request_interval: float = 1,
        n_retries: int = 3,
        retry_time: float = 3,
    ):
        self.logger = get_logger("googlemapsapi")
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

        self.driver.close()
        self.driver.quit()

        return True

    def _parse_url_to_feature_id(self, url: str) -> str:
        return re.findall("0[xX][0-9a-fA-F]+:0[xX][0-9a-fA-F]+", url)[0]

    def _parse_sort_by(self, sort_by: str) -> int:
        """Default to newest, enum=1"""
        return sort_by_enum.get(sort_by, 1)

    def _get_api_response(
        self,
        feature_id: str,
        async_: str = "",
        hl: str = "",
        sort_by_id: int = "",
        associated_topic: str = "",
        token: str = "",
    ) -> str:
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
        response = requests.get(query)
        return response.content.decode(encoding="unicode_escape")

    def _cut_response(self, text: str) -> str:
        idx_first_div = text.find("<div ")
        m = re.search("</div>", text, flags=re.REVERSE)
        idx_last_div = m.span()[1]
        text = text[idx_first_div:idx_last_div]
        return "<html><body>" + text + "</body></html>"

    def _get_reviews(
        self,
        response: str,
    ) -> Tuple[List[BeautifulSoup], str, int]:
        # Cut response to remove css
        response = self._cut_response(response)
        # Send page to soup
        # soup = BeautifulSoup(response, "html.parser")
        soup = BeautifulSoup(response, "lxml")
        dom = etree.HTML(soup.text)
        tree = html.document_fromstring(response)

        # Encontrando número de reviews e token de próxima página
        metadata_node = tree.xpath("//*[@data-google-review-count]")[0]
        review_count = int(metadata_node.attrib["data-google-review-count"])
        next_token = metadata_node.attrib["data-next-page-token"]

        # Iterando sobre texto de cada review
        reviews_tree = tree.xpath("/html/body/div[1]/div/div[2]/div[4]/div/div[2]/div")
        reviews_soup = [soup.find("div", dict(r.attrib)) for r in reviews_tree]

        # with open("examples/response_sorted.html", "w", encoding="utf-8") as f:
        #     f.writelines(soup.__str__())
        # with open("examples/response_sorted_review0.html", "w", encoding="utf-8") as f:
        #     f.writelines(reviews_soup[0].__str__())
        # with open("examples/response_sorted_review1.html", "w", encoding="utf-8") as f:
        #     f.writelines(reviews_soup[1].__str__())
        # with open("examples/response_sorted_review2.html", "w", encoding="utf-8") as f:
        #     f.writelines(reviews_soup[2].__str__())
        # with open("examples/response_sorted_review3.html", "w", encoding="utf-8") as f:
        #     f.writelines(reviews_soup[3].__str__())
        # with open("examples/response_sorted_review4.html", "w", encoding="utf-8") as f:
        #     f.writelines(reviews_soup[4].__str__())
        # with open("examples/response_sorted_review5.html", "w", encoding="utf-8") as f:
        #     f.writelines(reviews_soup[5].__str__())
        return reviews_soup, next_token, review_count

    def _parse_review(self, review: Tag):
        result = review_default_result.copy()

        # Parse text
        try:
            # text_block = review.find(True, jscontroller="MZnM8e")
            # if text_block.find(True, class_="review-full-text"):
            #     result["text"] = text_block.next_element.next_element(text=True)[0]
            # else:
            #     result["text"] = text_block.next_element(text=True)[0]

            text_block = review.find(True, class_="review-full-text")
            if not text_block:
                text_block = review.find(True, {"data-expandable-section": True})
            if text_block and isinstance(text_block.contents[0], NavigableString):
                text_block = text_block.text
                result["text"] = re.sub("\s", " ", text_block)
        except Exception as e:
            self.logger.error("error parsing text")
            self.logger.exception(e)
        # self.logger.info(f"text:{result['text']}")

        # Parse review rating
        try:
            rating_text = review.find(True, class_="Fam1ne EBe2gf").get("aria-label")
            rating_text = re.sub(",", ".", rating_text)
            rating = re.findall("[0-9]+[.][0-9]*", rating_text)
            result["rating"] = float(rating[0])
            result["rating_max"] = float(rating[1])
        except Exception as e:
            self.logger.error("error parsing rating")
            self.logger.exception(e)

        # Parse other ratings
        try:
            other_ratings = review.find(True, class_="k8MTF")
            if other_ratings:
                result["other_ratings"] = "".join(
                    [s for s in other_ratings.stripped_strings]
                )
        except Exception as e:
            self.logger.error("error parsing other_ratings")
            self.logger.exception(e)

        # Parse relative date
        try:
            result["relative_date"] = review.find(True, class_="dehysf lTi8oc").text
        except Exception as e:
            self.logger.error("error parsing relative_date")
            self.logger.exception(e)

        # Parse user name
        try:
            result["user_name"] = review.find(True, class_="TSUbDb").text
        except Exception as e:
            self.logger.error("error parsing user_name")
            self.logger.exception(e)

        # Parse user metadata
        try:
            user_node = review.find(True, class_="Msppse")
            result["user_url"] = user_node.get("href")
            result["user_is_local_guide"] = (
                True if user_node.find(True, class_="QV3IV") else False
            )
            user_comments = re.findall("[0-9]+(?= comentários)", user_node.text)
            if len(user_comments) == 1:
                result["user_comments"] = int(user_comments[0])
            user_photos = re.findall("[0-9]+(?= fotos)", user_node.text)
            if len(user_photos) == 1:
                result["user_photos"] = int(user_photos[0])
        except Exception as e:
            self.logger.error("error parsing user metadata")
            self.logger.exception(e)

        # Parse review id
        try:
            # result["review_id"] = review.find(True, {"data-ri": True}).get("data-ri")
            review_id = review.find(True, class_="RvU3D").get("href")
            result["review_id"] = re.findall("(?<=postId=).*?(?=&)", review_id)[0]
        except Exception as e:
            self.logger.error("error parsing review_id")
            self.logger.exception(e)

        # Make timestamp
        result["retrieval_date"] = str(datetime.now())

        return result

    def _node_to_xpath(self, node):
        node_type = {
            Tag: getattr(node, "name"),
            Comment: "comment()",
            NavigableString: "text()",
        }
        same_type_siblings = list(
            node.parent.find_all(
                lambda x: getattr(node, "name", True) == getattr(x, "name", False),
                recursive=False,
            )
        )
        if len(same_type_siblings) <= 1:
            return node_type[type(node)]
        pos = same_type_siblings.index(node) + 1
        return f"{node_type[type(node)]}[{pos}]"

    def _get_node_xpath(self, node: Union[Tag, Comment]):
        xpath = "/"
        elements = [f"{self._node_to_xpath(node)}"]
        for p in node.parents:
            if p.name == "[document]":
                break
            elements.insert(0, self._node_to_xpath(p))

        xpath = "/" + xpath.join(elements)
        return xpath

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
        url_name = re.findall("(?<=place/).*?(?=/)", url)[0]
        url_name = urllib.parse.unquote_plus(url_name)
        self.logger.info(f"Scraping url: {url_name}")

        feature_id = self._parse_url_to_feature_id(url)
        sort_by_id = self._parse_sort_by(sort_by)

        results = []
        j = 0

        n_requests = math.ceil((n_reviews) / 10)
        for i in range(n_requests):
            self.logger.info(f"Request: {i:>8}; review: {j:>8}")
            n = self.n_retries
            while n > 0:
                try:
                    response = self._get_api_response(
                        feature_id,
                        hl=hl,
                        sort_by_id=sort_by_id,
                        token=token,
                    )
                    reviews, token, review_count = self._get_reviews(response)
                    break
                except Exception as e:
                    self.logger.error(f"error on request: {i}")
                    self.logger.exception(e)
                    self.logger.info(f"waiting {self.retry_time} seconds")
                    time.sleep(self.retry_time)
                    n -= 1

            try:
                for review in reviews:
                    self.logger.info(f"Parsing review: {j:>8}")
                    result = self._parse_review(review)
                    result["token"] = token

                    writer.writerow(result.values())
                    file.flush()

                    results.append(result)
                    # self.logger.info(results)
                    j += 1
            except Exception as e:
                self.logger.error(f"error parsing request: {i}")
                self.logger.exception(e)

            # Waiting so google wont block this ip
            time.sleep(self.request_interval)

        self.logger.info(
            "Done Scraping Reviews\n"
            f"Requests made: {n_requests}\n"
            f"Reviews parsed: {j}"
        )

        return results
