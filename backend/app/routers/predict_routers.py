from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/predict", tags=["ml"])

class RiskInput(BaseModel):
    lat: float
    lon: float
    temp_c: float
    humidity_pct: float
    rain_1h_mm: float

class RiskOutput(BaseModel):
    flood_risk: float
    drought_risk: float
    severity: str

@router.post("/risk", response_model=RiskOutput)
def predict_risk(data: RiskInput):
    # SentinelX AI Model
    flood_risk = min(1.0, data.rain_1h_mm * 0.5 + (100-data.humidity_pct)/200)
    drought_risk = max(0.0, (data.temp_c-35)/10)
    severity = "🚨 HIGH" if flood_risk > 0.4 else "✅ LOW"
    
    return RiskOutput(
        flood_risk=round(flood_risk, 2),
        drought_risk=round(drought_risk, 2),
        severity=severity
    )
