import requests
from typing import Dict, Any

def get_current_weather(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return {}

def preprocess_weather_data(raw: Dict[str, Any]) -> Dict[str, float]:
    if not raw or "main" not in raw:
        raise ValueError("Invalid weather data")
    return {
        "temp": float(raw["main"]["temp"]),
        "humidity": float(raw["main"]["humidity"])
    }
