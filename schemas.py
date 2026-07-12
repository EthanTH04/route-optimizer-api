from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CityInput(BaseModel):
    """A single city sent by the user in a solve request."""
    name: str = Field(..., min_length=1, max_length=100)
    x: float
    y: float
    prize: int = Field(..., ge=0)


class SolveRequest(BaseModel):
    """The full request body for POST /solve."""
    cities: list[CityInput] = Field(..., min_length=3, max_length=100)
    budget: float = Field(..., gt=0)
    start_city: str = Field(..., min_length=1)
    end_city: str = Field(..., min_length=1)
    trials: int = Field(default=1000, ge=100, le=10000)


class AlgorithmResult(BaseModel):
    """Result from a single algorithm run."""
    algorithm: str
    num_cities: int
    budget: float
    total_distance: float
    prize_collected: int
    runtime_ms: float
    remaining_budget: float
    route: list[str]


class SolveResponse(BaseModel):
    """Full response from POST /solve - both algorithms plus comparison."""
    q_learning: AlgorithmResult
    p_marl: AlgorithmResult
    winner_by_prize: str
    winner_by_runtime: str
    q_learning_run_id: int
    p_marl_run_id: int


class RunSummary(BaseModel):
    """Summary of a stored algorithm run for the list endpoint."""
    id: int
    algorithm: str
    num_cities: int
    budget: float
    total_distance: float
    prize_collected: int
    runtime_seconds: float
    created_at: datetime

    class Config:
        from_attributes = True


class RunDetail(BaseModel):
    """Full details of a single stored run."""
    id: int
    algorithm: str
    num_cities: int
    budget: float
    total_distance: float
    prize_collected: int
    runtime_seconds: float
    route: str
    created_at: datetime

    class Config:
        from_attributes = True

class ExplainRequest(BaseModel):
    """Request body for POST /explain — takes two run IDs."""
    q_learning_run_id: int
    p_marl_run_id: int


class ExplainResponse(BaseModel):
    """Response for POST /explain — the AI-generated explanation."""
    explanation: str
    q_learning_run_id: int
    p_marl_run_id: int

class UserCreate(BaseModel):
    """Request body for POST /auth/register."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)


class UserResponse(BaseModel):
    """Response body for user info."""
    id: int
    username: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str