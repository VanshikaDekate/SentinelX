# backend/app/utils/preprocess.py

from typing import Dict, Any


def preprocess_openweather(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take the raw JSON from OpenWeather and return a simplified dict.
    Raises ValueError if OpenWeather returned an error (no 'main' data).
    """

    # Handle OpenWeather error responses like:
    # {"cod": 401, "message": "Invalid API key"} or {"cod": "400", "message": "..."}
    cod = str(raw.get("cod", "200"))
    if cod != "200":
        msg = raw.get("message", "OpenWeather returned an error")
        raise ValueError(f"OpenWeather error (cod={cod}): {msg}")

    main = raw.get("main", {})
    wind = raw.get("wind", {})
    rain = raw.get("rain", {})
    weather_list = raw.get("weather", [])

    temp_c = main.get("temp")
    humidity_pct = main.get("humidity")
    rain_1h_mm = rain.get("1h", 0.0)
    rain_3h_mm = rain.get("3h", 0.0)
    description = weather_list[0]["description"] if weather_list else None

    is_heavy_rain = rain_1h_mm >= 10.0
    is_very_humid = humidity_pct is not None and humidity_pct >= 80

    return {
        "temperature_c": temp_c,
        "humidity_pct": humidity_pct,
        "rain_1h_mm": rain_1h_mm,
        "rain_3h_mm": rain_3h_mm,
        "description": description,
        "is_heavy_rain": is_heavy_rain,
        "is_very_humid": is_very_humid,
    }

