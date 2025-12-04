import requests


def fetch_seeds():
    """
    Fetch seeds information from the API

    Returns:
        dict: JSON response from the API or None if request fails
    """
    url = "https://gagapi.onrender.com/seeds"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching seeds: {e}")
        return None


def fetch_gear():
    """
    Fetch gear information from the API

    Returns:
        dict: JSON response from the API or None if request fails
    """
    url = "https://gagapi.onrender.com/gear"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching gear: {e}")
        return None


def fetch_cosmetics():
    """
    Fetch cosmetics information from the API

    Returns:
        dict: JSON response from the API or None if request fails
    """
    url = "https://gagapi.onrender.com/cosmetics"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching cosmetics: {e}")
        return None


def fetch_eggs():
    """
    Fetch eggs information from the API

    Returns:
        dict: JSON response from the API or None if request fails
    """
    url = "https://gagapi.onrender.com/eggs"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching eggs: {e}")
        return None


def fetch_eventshop():
    """
    Fetch event shop information from the API

    Returns:
        dict: JSON response from the API or None if request fails
    """
    url = "https://gagapi.onrender.com/eventshop"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching event shop: {e}")
        return None


def fetch_weather():
    """
    Fetch weather information from the API

    Returns:
        dict: JSON response from the API or None if request fails
    """
    url = "https://gagapi.onrender.com/weather"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather: {e}")
        return None