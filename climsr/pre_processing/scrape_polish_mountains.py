# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import pandas as pd


def scrape_url1():
    response = requests.get(URL1)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p", attrs={"style": "text-align: center;"})

    current_mr = ""
    records = []
    for p in paragraphs:
        mountain_range_span = p.find("span", attrs={"style": "font-size: 14pt;"})
        mountain_peak_gps = p.text.split("GPS - ")

        if mountain_range_span:
            current_mr = mountain_range_span.text

        if len(mountain_peak_gps) > 1:
            lat, lon = tuple(map(float, mountain_peak_gps[-1].split(", ")))

            mountain_peak_split = mountain_peak_gps[0].split(" ")
            mountain_peak_alt = mountain_peak_split[-3]
            if mountain_peak_alt == "m":
                mountain_peak_alt = mountain_peak_split[-4]

            mountain_peak_name = " ".join(mountain_peak_split[:-3]).replace(
                mountain_peak_alt, ""
            )

            records.append(
                (current_mr, mountain_peak_name, lat, lon, mountain_peak_alt)
            )

    df = pd.DataFrame(
        records,
        columns=["mountain_range", "mountain_peak_name", "lat", "lon", "altitude"],
    )
    df.to_csv("../../datasets/mountain_peaks.csv", index=False, header=True)


if __name__ == "__main__":
    URL1 = "https://klubpodroznikow.com/relacje/polska/polska-gory/2690-najwyzsze-szczyty-w-polsce"
    # URL2 = "https://pl.wikipedia.org/wiki/Lista_najwyższych_szczytów_górskich_w_Polsce"

    scrape_url1()
