from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import Depends
from fastapi.security import HTTPBearer

SQLALCHEMY_DATABASE_URL = "sqlite:///./weather.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

security = HTTPBearer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_api_key(credentials: HTTPBearer = Depends(security)) -> str:
    return "83a3abc5014e2019b9e50e4aedb9c91a"
