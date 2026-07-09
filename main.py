from fastapi import FastAPI
from database import engine
import models

# Create all database tables defined in models.py
models.Base.metadata.create_all(bind=engine)

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "TSP Optimization API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/info")
def api_info():
    return {
        "name": "TSP Optimization API",
        "version": "0.1.0",
        "description": "Compares Q-Learning and P-MARL algorithms for the Budget-Constrained Traveling Salesman Problem"
    }