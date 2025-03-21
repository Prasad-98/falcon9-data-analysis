import datetime
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class SpaceXDataExtractor:
    """Extract and process SpaceX launch data from the API.

    This class fetches launch data from the SpaceX API, retrieves
    associated details for launchpads, payloads, rockets, and cores,
    and processes the data into a structured Pandas DataFrame.
    """

    SPACEX_LAUNCHES_URL: str = "https://api.spacexdata.com/v4/launches/past"
    LAUNCHPADS_URL: str = "https://api.spacexdata.com/v4/launchpads/"
    PAYLOADS_URL: str = "https://api.spacexdata.com/v4/payloads/"
    ROCKETS_URL: str = "https://api.spacexdata.com/v4/rockets/"
    CORES_URL: str = "https://api.spacexdata.com/v4/cores/"

    def __init__(self, limit_date: datetime.date = datetime.date(2020, 11, 13)) -> None:
        """
        Initialize the extractor with a limit date for launches.

        Args:
            limit_date (datetime.date): Only launches on or before this
                date will be processed.
        """
        self.limit_date: datetime.date = limit_date

        # Data containers for launch details
        self.booster_versions: List[Optional[str]] = []
        self.payload_masses: List[Optional[float]] = []
        self.orbits: List[Optional[str]] = []
        self.launch_sites: List[Optional[str]] = []
        self.outcomes: List[Optional[str]] = []
        self.flights: List[Optional[int]] = []
        self.grid_fins: List[Optional[bool]] = []
        self.reused_status: List[Optional[bool]] = []
        self.legs: List[Optional[bool]] = []
        self.landing_pads: List[Optional[str]] = []
        self.block_numbers: List[Optional[int]] = []
        self.reuse_counts: List[Optional[int]] = []
        self.core_serials: List[Optional[str]] = []
        self.longitudes: List[Optional[float]] = []
        self.latitudes: List[Optional[float]] = []

        # Main DataFrame containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None

    @staticmethod
    def fetch_json(url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch JSON data from the given URL with error handling.

        Args:
            url (str): The API URL to fetch data from.

        Returns:
            Optional[Dict[str, Any]]: The JSON response as a dictionary,
            or None if an error occurs.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            logging.error("Error fetching data from %s: %s", url, error)
            return None

    def fetch_launch_data(self) -> None:
        """
        Fetch launch data from the SpaceX API and preprocess the DataFrame.

        The method filters columns, removes rows with multiple cores or
        payloads, extracts single values from lists, and filters data
        based on the launch date.
        """
        response = self.fetch_json(self.SPACEX_LAUNCHES_URL)
        if response is None:
            logging.error("Failed to retrieve launch data.")
            return

        df = pd.json_normalize(response)
        logging.info("Columns in the raw data: %s", list(df.columns))

        # Keep only required columns
        df = df[['rocket', 'payloads', 'launchpad', 'cores',
                 'flight_number', 'date_utc']]

        # Remove rows with multiple cores or payloads
        df = df[df['cores'].map(len) == 1]
        df = df[df['payloads'].map(len) == 1]

        # Extract single values from lists
        df['cores'] = df['cores'].map(lambda x: x[0])
        df['payloads'] = df['payloads'].map(lambda x: x[0])

        # Convert date_utc to datetime.date and filter based on limit_date
        df['date'] = pd.to_datetime(df['date_utc']).dt.date
        df = df[df['date'] <= self.limit_date]

        self.raw_data = df.reset_index(drop=True)

    def get_launch_site_data(self) -> None:
        """Populate launch site related fields from the API."""
        if self.raw_data is None:
            logging.error("No launch data available for launch sites.")
            return

        for launchpad_id in self.raw_data['launchpad']:
            if launchpad_id:
                url = self.LAUNCHPADS_URL + str(launchpad_id)
                response = self.fetch_json(url)
                if response:
                    self.longitudes.append(response.get('longitude'))
                    self.latitudes.append(response.get('latitude'))
                    self.launch_sites.append(response.get('name'))
                else:
                    self.longitudes.append(None)
                    self.latitudes.append(None)
                    self.launch_sites.append(None)
            else:
                self.longitudes.append(None)
                self.latitudes.append(None)
                self.launch_sites.append(None)

    def get_payload_data(self) -> None:
        """Populate payload related fields from the API."""
        if self.raw_data is None:
            logging.error("No launch data available for payloads.")
            return

        for payload_id in self.raw_data['payloads']:
            if payload_id:
                url = self.PAYLOADS_URL + str(payload_id)
                response = self.fetch_json(url)
                if response:
                    self.payload_masses.append(response.get('mass_kg'))
                    self.orbits.append(response.get('orbit'))
                else:
                    self.payload_masses.append(None)
                    self.orbits.append(None)
            else:
                self.payload_masses.append(None)
                self.orbits.append(None)

    def get_booster_version_data(self) -> None:
        """Populate booster version information using the rocket ID."""
        if self.raw_data is None:
            logging.error("No launch data available for booster versions.")
            return

        for rocket_id in self.raw_data['rocket']:
            if rocket_id:
                url = self.ROCKETS_URL + str(rocket_id)
                response = self.fetch_json(url)
                if response:
                    self.booster_versions.append(response.get('name'))
                else:
                    self.booster_versions.append(None)
            else:
                self.booster_versions.append(None)

    def get_core_data(self) -> None:
        """Populate core related fields from the API."""
        if self.raw_data is None:
            logging.error("No launch data available for core information.")
            return

        for core_info in self.raw_data['cores']:
            core_id = core_info.get('core')
            if core_id:
                url = self.CORES_URL + str(core_id)
                response = self.fetch_json(url)
                if response:
                    self.block_numbers.append(response.get('block'))
                    self.reuse_counts.append(response.get('reuse_count'))
                    self.core_serials.append(response.get('serial'))
                else:
                    self.block_numbers.append(None)
                    self.reuse_counts.append(None)
                    self.core_serials.append(None)
            else:
                self.block_numbers.append(None)
                self.reuse_counts.append(None)
                self.core_serials.append(None)

            landing_success = core_info.get('landing_success')
            landing_type = core_info.get('landing_type')
            self.outcomes.append(f"{landing_success} {landing_type}")
            self.flights.append(core_info.get('flight'))
            self.grid_fins.append(core_info.get('gridfins'))
            self.reused_status.append(core_info.get('reused'))
            self.legs.append(core_info.get('legs'))
            self.landing_pads.append(core_info.get('landpad'))

    def process_all_data(self) -> None:
        """
        Execute all steps to fetch and process the launch data.

        The method calls data fetching methods and creates the final
        DataFrame from the collected data.
        """
        self.fetch_launch_data()
        if self.raw_data is None or self.raw_data.empty:
            logging.error("No data available to process.")
            return

        self.get_booster_version_data()
        self.get_launch_site_data()
        self.get_payload_data()
        self.get_core_data()
        self._create_dataframe()

    def _create_dataframe(self) -> None:
        """
        Construct the final DataFrame from the collected data.

        Rows with a booster version of 'Falcon 1' are filtered out.
        """
        launch_data = {
            'FlightNumber': list(self.raw_data['flight_number']),
            'Date': list(self.raw_data['date_utc']),
            'BoosterVersion': self.booster_versions,
            'PayloadMass': self.payload_masses,
            'Orbit': self.orbits,
            'LaunchSite': self.launch_sites,
            'Outcome': self.outcomes,
            'Flights': self.flights,
            'GridFins': self.grid_fins,
            'Reused': self.reused_status,
            'Legs': self.legs,
            'LandingPad': self.landing_pads,
            'Block': self.block_numbers,
            'ReusedCount': self.reuse_counts,
            'Serial': self.core_serials,
            'Longitude': self.longitudes,
            'Latitude': self.latitudes
        }
        df = pd.DataFrame(launch_data)
        self.processed_data = df[df['BoosterVersion'] != 'Falcon 1']

    def export_to_csv(self, filename: str = "SpaceX_API_data.csv") -> None:
        """
        Export the processed DataFrame to a CSV file.

        Args:
            filename (str): The name of the CSV file.
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(filename, index=False)
            logging.info("Data exported to %s", filename)
        else:
            logging.error("No processed data available. Run process_all_data() first.")


def main() -> None:
    """Run the SpaceX data extraction and export process."""
    extractor = SpaceXDataExtractor()
    extractor.process_all_data()
    extractor.export_to_csv()


if __name__ == "__main__":
    main()