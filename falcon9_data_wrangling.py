import logging
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class Falcon9Scraper:
    """
    Scrape Falcon 9 launch data from Wikipedia and export to CSV.

    The class fetches the HTML content from the Wikipedia page,
    parses the Falcon 9 launch table, cleans the data, and exports
    it to a CSV file.
    """

    URL: str = "https://en.wikipedia.org/wiki/List_of_Falcon_9_launches"

    def get_html_content(self, url: str) -> Optional[str]:
        """
        Send a GET request to the specified URL and return its HTML content.

        Args:
            url (str): URL to fetch the HTML from.

        Returns:
            Optional[str]: The HTML content if successful, else None.

        Raises:
            HTTPError: If the HTTP request fails.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            logging.error("Error fetching HTML content from %s: %s", url, error)
            return None

    def parse_launch_table(self, html: str) -> pd.DataFrame:
        """
        Parse the HTML content to extract Falcon 9 launch data from the table.

        Args:
            html (str): HTML content of the Wikipedia page.

        Returns:
            pd.DataFrame: DataFrame containing the parsed launch data.
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        if table is None:
            logging.error("No table with class 'wikitable' found.")
            return pd.DataFrame()

        # Extract header names
        headers = [
            th.get_text(separator=" ", strip=True)
            for th in table.find("tr").find_all("th")
        ]

        data_rows = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            row_text = [cell.get_text(separator=" ", strip=True) for cell in cells]

            # Skip rows that don't match header structure or contain too much text.
            if len(row_text) < len(headers) or len(" ".join(row_text)) > 200:
                continue

            if len(row_text) < len(headers):
                row_text += [None] * (len(headers) - len(row_text))

            data_rows.append(dict(zip(headers, row_text)))

        df = pd.DataFrame(data_rows)
        return df

    def clean_launch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns and perform basic cleaning on the DataFrame.

        The function maps the scraped column names to the desired names:
        Flight No., Launch site, Payload, PayloadMass, Orbit, Customer,
        Launch outcome, Version Booster, Booster landing, Date, Time.

        Args:
            df (pd.DataFrame): Raw DataFrame from scraped data.

        Returns:
            pd.DataFrame: Cleaned DataFrame with renamed columns.
        """
        column_mapping = {
            "Flight": "Flight No.",
            "Launch site": "Launch site",
            "Payload": "Payload",
            "Payload mass": "PayloadMass",
            "Orbit": "Orbit",
            "Customer": "Customer",
            "Launch outcome": "Launch outcome",
            "Booster": "Version Booster",
            "Booster landing": "Booster landing",
            "Date": "Date",
            "Time": "Time"
        }
        # Only map columns that exist in the DataFrame to avoid KeyError
        df_clean = df.rename(
            columns={key: value for key, value in column_mapping.items()
                     if key in df.columns}
        )
        return df_clean

    def export_data_to_csv(self, df: pd.DataFrame,
                           filename: str = "spaceX_web_scraped.csv") -> None:
        """
        Export the given DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to export.
            filename (str, optional): Filename for the CSV file.
                Defaults to "spaceX_web_scraped.csv".
        """
        try:
            df.to_csv(filename, index=False)
            logging.info("Data exported successfully to %s", filename)
        except Exception as error:
            logging.error("Error exporting data to CSV: %s", error)

    def run(self) -> None:
        """
        Execute the scraping, cleaning, and export of Falcon 9 launch data.
        """
        html_content = self.get_html_content(self.URL)
        if html_content is None:
            logging.error("Failed to retrieve HTML content.")
            return

        df_raw = self.parse_launch_table(html_content)
        if df_raw.empty:
            logging.error("No data extracted from the HTML content.")
            return

        df_clean = self.clean_launch_data(df_raw)
        self.export_data_to_csv(df_clean)


def main() -> None:
    """Main function to execute the Falcon 9 scraper."""
    scraper = Falcon9Scraper()
    scraper.run()


if __name__ == "__main__":
    main()