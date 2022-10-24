# https://stackoverflow.com/questions/53749984/selenium-python-unable-to-scroll-down-while-fetching-google-reviews
# https://hackernoon.com/scraping-google-maps-reviews
# https://gist.github.com/gruentee/203e4ba3791581070df9a4b1e6c55549
# https://beautiful-soup-4.readthedocs.io/en/latest/index.html?highlight=find#attrs
# https://github.com/emerson-matos/googlemaps-scraper/blob/main/monitor.py
# https://gist.github.com/IanHopkinson/ad45831a2fb73f537a79

import requests
from lxml import etree, html
from bs4 import BeautifulSoup
import regex as re
from src.scrapper import get_text, cut_text, get_node_xpath, node_to_xpath

feature_id = "0x94ce03043613e3d9:0x72a1063f1eb9c819"


def scrape_reviews():
    # Realizando post na api de reviews do front-end
    token = ""
    text = get_text(feature_id, token)
    print("text[:100]\n", text[:100])

    # Removendo css
    text = cut_text(text)
    print("text[:100]\n", text[:100])

    # Salvando em arquivo
    with open("text.html", "w", encoding="utf-8") as f:
        f.writelines(text)

    # Iniciando soup e etree
    soup = BeautifulSoup(text, "lxml")
    dom = etree.HTML(soup.text)
    tree = html.document_fromstring(text)
    # tree = etree.fromstring(text)

    # Encontrando paths das tags da class review-full-text
    node = soup.find(class_="review-full-text")
    print("node.text\n", node.text)
    nodes = soup.find_all(class_="review-full-text")
    paths = [get_node_xpath(n) for n in nodes]
    [print(path) for path in paths]

    # Iterando sobre texto de cada review
    print("\n")
    for i in range(1, 11):
        print(f"div[{i}] text_content:")
        t = tree.xpath(f"/html/body/div[1]/div/div[2]/div[4]/div/div[2]/div[{i}]")
        t = t[0].text_content()
        print(t, "\n")

    # Encontrando número de reviews e token de próxima página
    e = tree.xpath("//*[@data-google-review-count]")[0]
    print("data-google-review-count:", e.attrib["data-google-review-count"])
    token = e.attrib["data-next-page-token"]
    print("data-next-page-token:", token)

    # # Encontrando primeiro texto
    # e = tree.xpath('//span[@class="review-full-text"]')[0]
    # print("text:", e.text)
    # es = tree.xpath('//span[@aria-label=contains(.,"Classificado como ")]')

    print("\n ________ next token \n")

    text = get_text(feature_id, token)
    text = cut_text(text)
    with open("text_next.html", "w", encoding="utf-8") as f:
        f.writelines(text)

    # Iniciando soup e etree
    soup = BeautifulSoup(text, "lxml")
    tree = html.document_fromstring(text)

    # Iterando sobre texto de cada review
    print("\n")
    for i in range(1, 11):
        print(f"div[{i}] text_content:")
        t = tree.xpath(f"/html/body/div[1]/div/div[2]/div[4]/div/div[2]/div[{i}]")
        t = t[0].text_content()
        print(t, "\n")


if __name__ == "__main__":
    scrape_reviews()
