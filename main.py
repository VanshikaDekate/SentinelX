from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import sqlite3
from datetime import datetime
import os
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SentinelX - Global Flood AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OWM_API_KEY = "83a3abc5014e2019b9e50e4aedb9c91a"
DB_PATH = "weather.db"

def predict_risk(temp, hum, rain):
    score = (temp/50 + hum/100 + rain/10) / 3
    if score > 0.7: return score, "🔴 HIGH RISK"
    elif score > 0.4: return score, "🟡 MEDIUM RISK"
    return score, "🟢 LOW RISK"

@app.get("/api/search/{query}")
async def search_cities(query: str):
    url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=5"
    try:
        resp = requests.get(url, headers={'User-Agent': 'SentinelX/1.0'})
        cities = resp.json()
        return {
            "query": query,
            "results": [{"name": c['display_name'].split(',')[0], 
                        "lat": float(c['lat']), 
                        "lon": float(c['lon'])} for c in cities if c.get('lat')]
        }
    except:
        return {"results": []}

@app.get("/api/weather/{lat}/{lon}")
async def get_weather(lat: float, lon: float):
    params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}
    resp = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params)
    data = resp.json()
    
    city = data.get("name", "Unknown")
    temp = data["main"]["temp"]
    hum = data["main"]["humidity"]
    rain = data.get("rain", {}).get("1h", 0)
    
    score, risk = predict_risk(temp, hum, rain)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS weather (lat REAL, lon REAL, city TEXT, temp REAL, hum INTEGER, rain REAL, risk TEXT, score REAL, time TEXT)''')
    c.execute('INSERT INTO weather VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', 
              (lat, lon, city, temp, hum, rain, risk, score, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return {
        "city": city, "lat": lat, "lon": lon,
        "temp_c": round(temp, 1), "humidity": hum, "rain_1h": rain,
        "risk_score": round(score, 2), "flood_risk": risk
    }

@app.get("/api/precise/{lat}/{lon}")
async def precise(lat: float, lon: float):
    return await get_weather(lat, lon)

@app.get("/")
async def dashboard():
    return HTMLResponse('''
<!DOCTYPE html>
<html><head><title>SentinelX 🌍</title>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" rel="stylesheet">
<style>body{background:linear-gradient(135deg,#1e3a8a 0%,#7c3aed 50%,#dc2626 100%);min-height:100vh;padding:2rem;font-family:system-ui}.search-box{background:rgba(255,255,255,0.1);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.2);border-radius:2rem;padding:2rem;margin:2rem 0}.city-result{cursor:pointer;padding:1rem;margin:0.5rem 0;border-radius:1rem;background:rgba(255,255,255,0.15);transition:all 0.3s;font-size:1.2rem}.city-result:hover{background:rgba(255,255,255,0.3);transform:translateX(0.5rem);box-shadow:0 10px 30px rgba(0,0,0,0.3)}.risk-high{background:linear-gradient(45deg,#ef4444,#dc2626);color:white;padding:1rem 2rem;border-radius:50px;font-weight:bold;display:inline-block}.risk-medium{background:linear-gradient(45deg,#f59e0b,#d97706);color:white;padding:1rem 2rem;border-radius:50px;font-weight:bold;display:inline-block}.risk-low{background:linear-gradient(45deg,#10b981,#059669);color:white;padding:1rem 2rem;border-radius:50px;font-weight:bold;display:inline-block}#map{height:500px;border-radius:2rem;border:2px solid rgba(255,255,255,0.3);margin-top:2rem}</style></head>
<body class="text-white"><div class="max-w-4xl mx-auto">
<h1 class="text-6xl font-black text-center mb-12 bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 bg-clip-text text-transparent drop-shadow-2xl">SentinelX 🌍 FLOOD AI</h1>

<div class="search-box text-center">
<h2 class="text-4xl font-bold mb-8">🔍 Search ANY City Worldwide</h2>
<input id="citySearch" placeholder="Mumbai, London, Tokyo, Paris..." class="w-full p-6 text-2xl rounded-2xl bg-white/20 border-2 border-white/30 focus:border-yellow-400 focus:outline-none mb-6 text-white font-bold">
<div id="cityResults" class="max-h-64 overflow-y-auto"></div>
</div>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
<input id="lat" value="19.0760" step="0.0001" placeholder="Latitude" class="p-4 rounded-xl bg-white/10 border border-white/20 text-xl focus:border-yellow-400">
<input id="lon" value="72.8777" step="0.0001" placeholder="Longitude" class="p-4 rounded-xl bg-white/10 border border-white/20 text-xl focus:border-yellow-400">
<button onclick="getWeather()" class="p-4 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl text-xl font-bold hover:scale-105 shadow-2xl">🎯 GET WEATHER</button>
</div>

<div id="result" class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 hidden mb-8 shadow-2xl">
<h3 id="title" class="text-4xl font-bold mb-6 text-center"></h3>
<div id="data" class="grid grid-cols-2 gap-6 text-2xl"></div>
</div>

<div id="map"></div></div>

<script>
const map = L.map("map").setView([19.076,72.8777],10);L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);let currentMarker=null;
document.getElementById("citySearch").addEventListener("input",async e=>{const query=e.target.value;if(query.length<2){document.getElementById("cityResults").innerHTML="";return}const res=await fetch(`/api/search/${encodeURIComponent(query)}`);const data=await res.json();document.getElementById("cityResults").innerHTML=data.results.map(city=>`<div class="city-result" onclick="selectCity(${city.lat},${city.lon},'${city.name.replace(/'/g,"\\'")}')">🌍 ${city.name}</div>`).join("")||'<div class="city-result text-center opacity-50 py-8">No cities found</div>';});
function selectCity(lat,lon,name){document.getElementById("lat").value=lat.toFixed(4);document.getElementById("lon").value=lon.toFixed(4);document.getElementById("citySearch").value=name;document.getElementById("cityResults").innerHTML="";getWeather();}
async function getWeather(){const lat=parseFloat(document.getElementById("lat").value);const lon=parseFloat(document.getElementById("lon").value);const res=await fetch(`/api/weather/${lat}/${lon}`);const data=await res.json();document.getElementById("title").textContent=`${data.city} (${data.lat.toFixed(4)}, ${data.lon.toFixed(4)})`;document.getElementById("data").innerHTML=`<div>🌡️ ${data.temp_c}°C</div><div>💧 ${data.humidity}%</div><div>🌧️ ${data.rain_1h}mm</div><div class="risk-${data.flood_risk.includes('HIGH')?'high':data.flood_risk.includes('MEDIUM')?'medium':'low'}">${data.flood_risk}<br>Score: ${data.risk_score}</div>`;document.getElementById("result").classList.remove("hidden");if(currentMarker)map.removeLayer(currentMarker);const color=data.flood_risk.includes("HIGH")?"🔴":data.flood_risk.includes("MEDIUM")?"🟡":"🟢";currentMarker=L.marker([lat,lon],{icon:L.divIcon({html:color,className:"text-4xl font-bold"})}).addTo(map).bindPopup(`<b>${data.flood_risk}</b>`);map.setView([lat,lon],12);}
getWeather();
</script></body></html>''')

print("✅ SentinelX READY!")

# 🌦️ IMD INDIA WEATHER (Pune/Mumbai Stations)
@app.get("/api/imd/{city}")
async def imd_weather(city: str):
    # IMD Station IDs: Pune=43063, Mumbai=43003 (Santacruz), Mumbai Colaba=43057
    stations = {"pune": 43063, "mumbai": 43003, "mumbai2": 43057}
    
    if city.lower() not in stations:
        return {"error": "Use: pune, mumbai, mumbai2"}
    
    station_id = stations[city.lower()]
    try:
        # rtdtwo/india-weather-rest API (IMD data)
        resp = requests.get(f"https://rtdtwo.github.io/india-weather-rest/weather/{station_id}")
        imd_data = resp.json()
        
        # Extract key IMD data
        temp_max = imd_data["result"]["temperature"]["max"]["value"]
        temp_min = imd_data["result"]["temperature"]["min"]["value"]
        humidity_morn = imd_data["result"]["humidity"]["morning"]
        forecast = imd_data["result"]["forecast"][0]["condition"]
        
        # IMD Flood Risk Logic
        risk_score = (temp_max/50 + (100-humidity_morn)/100) / 2
        if "rain" in forecast.lower() or "thunderstorm" in forecast.lower():
            risk_score += 0.3
            imd_risk = "🟡 MEDIUM RISK (IMD Rain Alert)"
        elif risk_score > 0.6:
            imd_risk = "🔴 HIGH RISK (IMD Hot+Dry)"
        else:
            imd_risk = "🟢 LOW RISK (IMD)"
        
        return {
            "city": city.upper(), "station_id": station_id,
            "temp_max": round(temp_max, 1), "temp_min": round(temp_min, 1),
            "humidity_morning": humidity_morn,
            "forecast": forecast[:100] + "...",
            "imd_risk": imd_risk,
            "risk_score": round(risk_score, 2),
            "source": "IMD Official"
        }
    except:
        return {"error": "IMD API temp down", "fallback": "Use /api/weather Pune coords"}

@app.get("/api/india/{city}")
async def india_weather(city: str):
    # Get IMD data
    imd = await imd_weather(city)
    
    # Get OpenWeather (coords fallback)
    coords = {"pune": (18.5204, 73.8567), "mumbai": (19.0760, 72.8777)}
    if city.lower() in coords:
        lat, lon = coords[city.lower()]
        owm = await weather(lat, lon)
        return {
            "imd": imd,
            "openweather": owm,
            "recommended": "IMD" if "HIGH" in imd.get("imd_risk", "") else "OpenWeather"
        }
    return imd

