from sqlalchemy import Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class WeatherRecord(Base):
    __tablename__ = "weather_records"
    
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    temperature_c = Column(Float)
    humidity_pct = Column(Integer)
    rain_1h_mm = Column(Float, default=0.0)
    rain_3h_mm = Column(Float, default=0.0)
    recorded_at = Column(DateTime, default=datetime.utcnow)
