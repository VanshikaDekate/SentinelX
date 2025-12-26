import requests

def get_current_weather(lat: float, lon: float, api_key: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }
    resp = requests.get(url, params=params, timeout=10)
    return resp.json()
