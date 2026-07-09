from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class AlgorithmRun(Base):
    __tablename__ = "algorithm_runs"

    id = Column(Integer, primary_key=True, index=True)
    algorithm = Column(String, nullable=False)  # "q-learning" or "p-marl"
    num_cities = Column(Integer, nullable=False)
    budget = Column(Float, nullable=False)
    total_distance = Column(Float, nullable=False)
    prize_collected = Column(Float, nullable=False)
    runtime_seconds = Column(Float, nullable=False)
    route = Column(Text, nullable=False)  # Stored as JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())