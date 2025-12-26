from sqlalchemy import Column, Float, DateTime, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class WeatherRecord(Base):
    __tablename__ = "weather_records"
    
    record_id = Column(Integer, primary_key=True, autoincrement=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    temperature_c = Column(Float)
    humidity_pct = Column(Float)
    rain_1h_mm = Column(Float, default=0.0)
    rain_3h_mm = Column(Float, default=0.0)
    recorded_at = Column(DateTime)
