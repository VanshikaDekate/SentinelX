from fastapi import APIRouter, HTTPException
import os
import requests

router = APIRouter()

@router.get("/weather/")
def get_weather(lat: float, lon: float):
    """TEST OpenWeatherMap API ONLY"""
    
    api_key = "83a3abc5014e2019b9e50e4aedb9c91a"  # HARDCODED for test
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    
    print(f"🌐 Calling: {url} with lat={lat}, lon={lon}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"📡 Response status: {response.status_code}")
        print(f"📡 Response text: {response.text[:200]}...")
        
        data = response.json()
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"API Error {response.status_code}: {data}")
        
        return {
            "city": data.get("name", "Unknown"),
            "temp_c": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "rain_1h": data.get("rain", {}).get("1h", 0),
            "status": "✅ API WORKING!"
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
