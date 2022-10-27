# -*- coding: utf-8 -*-
import pandas as pd
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import datetime
import time
import traceback
import numpy as np
import itertools
import re

from src.customlogger import get_logger

GM_WEBPAGE = "https://www.google.com/maps/"
MAX_WAIT = 10
MAX_RETRY = 5
MAX_SCROLLS = 40


class GoogleMapsScraper:
    def __init__(self, debug=False, driver="firefox"):
        self.debug = debug
        self.logger = get_logger("googlemaps")
        if driver == "firefox":
            self.driver = self.__get_firefox_driver()
        elif driver == "chrome":
            self.driver = self.__get_chrome_driver()
        else:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            self.logger.exception(exc_value)

        self.driver.close()
        self.driver.quit()

        return True

    def sort_by(self, url, ind):
        self.driver.get(url)
        wait = WebDriverWait(self.driver, MAX_WAIT)

        # open dropdown menu
        clicked = False
        tries = 0
        while not clicked and tries < MAX_RETRY:
            try:
                # if not self.debug:
                #    menu_bt = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.cYrDcjyGO77__container')))
                # else:
                wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            '//button[@jsaction="pane.rating.moreReviews"] | //button[@jsaction="pane.reviewChart.moreReviews"]',
                        )
                    )
                ).click()

                menu_bt = wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            '//button[@aria-label="Mais relevantes"] | //button[@aria-label="Classificar avaliações"] | //button[@aria-label="Most relevant"] | //button[@aria-label="Sort reviews"]',
                        )
                    )
                )
                menu_bt.click()

                clicked = True
                time.sleep(3)
            except Exception as e:
                tries += 1
                self.logger.warn("Failed to click sorting button")

            # failed to open the dropdown
            if tries == MAX_RETRY:
                return -1

        #  element of the list specified according to ind
        recent_rating_bt = self.driver.find_elements(
            By.XPATH, "//li[@role='menuitemradio']"
        )[ind]
        recent_rating_bt.click()

        # wait to load review (ajax call)
        time.sleep(5)

        return 0

    def get_places(self, method="urls", keyword_list=None):

        df_places = pd.DataFrame()

        if method == "urls":
            # search_point_url = row['url']  # TODO:
            pass
        if method == "squares":
            search_point_url_list = self._gen_search_points_from_square(
                keyword_list=keyword_list
            )
        else:
            # search_point_url = f"https://www.google.com/maps/search/{row['keyword']}/@{str(row['longitude'])},{str(row['latitude'])},{str(row['zoom'])}z"
            # TODO:
            pass

        for i, search_point_url in enumerate(search_point_url_list):

            if (i + 1) % 10 == 0:
                print(f"{i}/{len(search_point_url_list)}")
                df_places = df_places[
                    [
                        "search_point_url",
                        "href",
                        "name",
                        "rating",
                        "num_reviews",
                        "close_time",
                        "other",
                    ]
                ]
                df_places.to_csv("output/places_wax.csv", index=False)

            try:
                self.driver.get(search_point_url)
            except NoSuchElementException:
                self.driver.quit()
                self.driver = self.__get_chrome_driver()
                self.driver.get(search_point_url)

            # Gambiarra to load all places into the page
            scrollable_div = self.driver.find_element(
                By.CSS_SELECTOR,
                "div.siAUzd-neVct.section-scrollbox.cYB2Ge-oHo7ed.cYB2Ge-ti6hGc > div[aria-label*='Results for']",
            )
            for i in range(10):
                self.driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div
                )

            # Get places names and href
            # time.sleep(2)
            response = BeautifulSoup(self.driver.page_source, "html.parser")
            div_places = response.select("div[jsaction] > a[href]")
            # print(len(div_places))
            for div_place in div_places:
                place_info = {
                    "search_point_url": search_point_url.replace(
                        "https://www.google.com/maps/search/", ""
                    ),
                    "href": div_place["href"],
                    "name": div_place["aria-label"],
                    "rating": None,
                    "num_reviews": None,
                    "close_time": None,
                    "other": None,
                }

                df_places = df_places.append(place_info, ignore_index=True)
        df_places = df_places[
            [
                "search_point_url",
                "href",
                "name",
                "rating",
                "num_reviews",
                "close_time",
                "other",
            ]
        ]
        df_places.to_csv("output/places_wax.csv", index=False)
        self.driver.quit()

    def _gen_search_points_from_square(self, keyword_list=None):
        # TODO: Generate search points from corners of square

        keyword_list = [] if keyword_list is None else keyword_list

        square_points = pd.read_csv("input/square_points.csv")

        cities = square_points["city"].unique()

        search_urls = []

        for city in cities:

            df_aux = square_points[square_points["city"] == city]
            latitudes = np.linspace(
                df_aux["latitude"].min(), df_aux["latitude"].max(), num=20
            )
            longitudes = np.linspace(
                df_aux["longitude"].min(), df_aux["longitude"].max(), num=20
            )
            coordinates_list = list(
                itertools.product(latitudes, longitudes, keyword_list)
            )

            search_urls += [
                f"https://www.google.com/maps/search/{coordinates[2]}/@{str(coordinates[1])},{str(coordinates[0])},{str(15)}z"
                for coordinates in coordinates_list
            ]

        return search_urls

    def get_reviews(self, writer, file, n_reviews=10):
        l = int((n_reviews + 1) / 10)

        parsed_reviews = []
        j_last = -1
        j = 0
        for i in range(l):
            if j_last == j:
                self.logger.info(f"Waiting...")
                time.sleep(2)
                self.__scroll()
            self.logger.info(f"Scrolling {i}/{l} parsed {j}/{n_reviews}")
            # scroll to load reviews
            self.__scroll()

            # wait for other reviews to load (ajax)
            time.sleep(1)

            # expand review text
            self.__expand_reviews()
            time.sleep(1)

            # send page to soup
            response = BeautifulSoup(self.driver.page_source, "html.parser")

            if i == 0:
                review = response.find("div", class_="jftiEf fontBodyMedium")
                # Parsing review
                self.logger.info(f"Parsing {j}/{n_reviews}")
                parsed = self.__parse(review)
                parsed_reviews.append(parsed)
                writer.writerow(parsed.values())
                file.flush()
                j += 1

            while True:
                # Selecting next review
                review = response.find("div", review.attrs)
                try:
                    review_next = review.find_next(
                        "div", class_="jftiEf fontBodyMedium"
                    )
                except Exception as e:
                    self.logger.info(f"Error getting next review: {e}")
                    break
                if review_next is None:
                    break

                # Parsing review
                self.logger.info(f"Parsing {j}/{n_reviews}")
                parsed = self.__parse(review_next)
                parsed_reviews.append(parsed)
                writer.writerow(parsed.values())
                file.flush()

                review = review_next
                j += 1
                j_last = j

        return parsed_reviews

    def get_account(self, url):

        self.driver.get(url)

        # ajax call also for this section
        time.sleep(4)

        resp = BeautifulSoup(self.driver.page_source, "html.parser")

        place_data = self.__parse_place(resp)

        # writer.writerow(place_data.values())

        return place_data

    def __parse(self, review):

        item = {}

        try:
            # TODO: Subject to changes
            id_review = review["data-review-id"]
        except Exception as e:
            self.logger.error("error parsing id_review")
            self.logger.exception(e)
            id_review = None

        try:
            # TODO: Subject to changes
            username = review["aria-label"]
        except Exception as e:
            self.logger.error("error parsing username")
            self.logger.exception(e)
            username = None

        try:
            # TODO: Subject to changes
            review_text = self.__filter_string(
                review.find("span", class_="wiI7pd").text
            )
        except Exception as e:
            self.logger.error("error parsing review_text")
            self.logger.exception(e)
            review_text = None

        try:
            # TODO: Subject to changes
            rating = float(review.find("span", class_="fzvQIb").text.split("/")[0])
        except Exception as e:
            self.logger.error("error parsing rating")
            self.logger.exception(e)
            rating = None

        try:
            # TODO: Subject to changes
            relative_date = review.find("span", class_="xRkPPb").text
            relative_date = re.sub("( on | em ).*", "", relative_date)
        except Exception as e:
            self.logger.error("error parsing relative_date")
            self.logger.exception(e)
            relative_date = None

        try:
            n_reviews_photos = (
                review.find("div", class_="section-review-subtitle")
                .find_all("span")[1]
                .text
            )
            metadata = n_reviews_photos.split("\xe3\x83\xbb")
            if len(metadata) == 3:
                n_photos = int(metadata[2].split(" ")[0].replace(".", ""))
            else:
                n_photos = 0

            idx = len(metadata)
            n_reviews = int(metadata[idx - 1].split(" ")[0].replace(".", ""))

        except Exception as e:
            n_reviews = 0
            n_photos = 0

        try:
            user_url = review.find("a")["href"]
        except Exception as e:
            self.logger.error("error parsing user_url")
            self.logger.exception(e)
            user_url = None

        item["id_review"] = id_review
        item["caption"] = review_text

        # depends on language, which depends on geolocation defined by Google Maps
        # custom mapping to transform into date should be implemented
        item["relative_date"] = relative_date

        # store datetime of scraping and apply further processing to calculate
        # correct date as retrieval_date - time(relative_date)
        item["retrieval_date"] = datetime.now()
        item["rating"] = rating
        item["username"] = username
        item["n_review_user"] = n_reviews
        item["n_photo_user"] = n_photos
        item["url_user"] = user_url

        return item

    def __parse_place(self, response: BeautifulSoup):

        place = {}

        try:
            place["place_name"] = (
                response.find("h1", class_="DUwDvf fontHeadlineLarge")
                .get_text()
                .strip()
            )
        except Exception as e:
            self.logger.error("error parsing place_name")
            self.logger.exception(e)
            place["place_name"] = ""

        try:
            stars_text = response.find(
                string=re.compile("hotel de [0-9] estrelas|[0-9]-star hotel")
            )
            stars = re.sub("^hotel de | estrelas$|-star hotel$", "", stars_text)
            place["hotel_stars"] = int(stars)
        except Exception as e:
            place["hotel_stars"] = 0

        try:
            place["overall_rating"] = float(
                response.find("div", class_="fontDisplayLarge").text.replace(",", ".")
            )
        except Exception as e:
            self.logger.error("error parsing overall_rating")
            self.logger.exception(e)
            place["overall_rating"] = "NOT FOUND"

        try:
            n_reviews_text = response.find(
                "button", class_="HHrUdb fontTitleSmall rqjGif"
            ).text
            n_reviews_text = re.sub("[.]|,| reviews| comentários", "", n_reviews_text)
            place["n_reviews"] = int(n_reviews_text)
        except Exception as e:
            self.logger.error("error parsing n_reviews")
            self.logger.exception(e)
            place["n_reviews"] = 0

        return place

    # expand review description
    def __expand_reviews(self):
        # use XPath to load complete reviews
        # TODO: Subject to changes
        links = self.driver.find_elements(
            By.XPATH, '//button[@jsaction="pane.review.expandReview"]'
        )
        for l in links:
            l.click()
            time.sleep(0.01)

    def __scroll(self):
        # TODO: Subject to changes
        scrollable_div = self.driver.find_element(
            By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf"
        )
        self.driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div
        )
        # self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def __get_chrome_driver(self):
        self.logger.info("iniciando webdriver")
        options = webdriver.ChromeOptions()
        options.binary_location = (
            "C:/Program Files/Google/Chrome/Application/chrome.exe"
        )
        self.logger.info(f"options.BinaryLocation {options.binary_location}")

        if not self.debug:
            options.add_argument("--headless")
        else:
            options.add_argument("--window-size=1366,768")

        options.add_argument("--disable-notifications")
        # options.add_argument("--lang=en-GB")
        options.add_argument("--lang=en-US")
        input_driver = webdriver.Remote(
            "http://localhost:4444/wd/hub", DesiredCapabilities.CHROME, options=options
        )
        self.logger.info("webdriver chrome carregado")

        # first lets click on google agree button so we can continue
        try:
            input_driver.get(GM_WEBPAGE)
            agree = WebDriverWait(input_driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//span[contains(text(), "I agree")]')
                )
            )
            agree.click()

            # back to the main page
            input_driver.switch_to_default_content()
        except:
            pass

        return input_driver

    def __get_firefox_driver(self):
        self.logger.info("iniciando webdriver")
        options = webdriver.FirefoxOptions()
        options.binary_location = "C:/Program Files/Mozilla Firefox/firefox.exe"
        self.logger.info(f"options.BinaryLocation {options.binary_location}")

        if not self.debug:
            options.add_argument("--headless")
        else:
            options.add_argument("--window-size=1366,768")

        options.add_argument("--disable-notifications")
        # options.add_argument("--lang=en-GB")
        # options.add_argument("--lang=en-US")
        options.set_preference("intl.accept_languages", "en-US")
        # input_driver = webdriver.Remote(
        #     "http://localhost:4444/wd/hub", DesiredCapabilities.FIREFOX, options=options
        # )
        input_driver = webdriver.Remote(
            "http://localhost:4444/wd/hub", options.to_capabilities()
        )
        self.logger.info("webdriver firefox carregado")

        # first lets click on google agree button so we can continue
        try:
            input_driver.get(GM_WEBPAGE)
            agree = WebDriverWait(input_driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//span[contains(text(), "I agree")]')
                )
            )
            agree.click()

            # back to the main page
            input_driver.switch_to_default_content()
        except:
            pass

        return input_driver

    # util function to clean special characters
    def __filter_string(self, s):
        s = re.sub("\s", " ", s)
        s = re.sub('"', "'", s)
        return s
