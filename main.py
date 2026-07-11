from fastapi import FastAPI
from database import engine
import json

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from database import engine, get_db
import models
from schemas import (
    SolveRequest,
    SolveResponse,
    AlgorithmResult,
    RunSummary,
    RunDetail,
)
from algorithms import prepare_cities
from algorithms.city import City
from algorithms.q_learning import QLearning
from algorithms.p_marl import PMARL


models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Route Optimizer API",
    description="Compares Q-Learning and P-MARL algorithms for the Budget-Constrained Traveling Salesman Problem",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"message": "Route Optimizer API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest, db: Session = Depends(get_db)):
    """
    Run both Q-Learning and P-MARL on the given cities and budget.
    Stores both runs in the database and returns a comparison.
    """
    # Convert Pydantic city input to internal City objects
    input_cities = [
        City(c.name, c.x, c.y, c.prize) for c in request.cities
    ]

    try:
        prepared_cities = prepare_cities(input_cities, request.start_city, request.end_city)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run Q-Learning
    q = QLearning(
        cities=prepared_cities,
        budget=request.budget,
        trials=request.trials,
    )
    q_result = q.run()

    # Run P-MARL
    p = PMARL(
        cities=prepared_cities,
        budget=request.budget,
        trials=request.trials,
    )
    p_result = p.run()

    # Store both runs in the database
    q_run = models.AlgorithmRun(
        algorithm=q_result["algorithm"],
        num_cities=q_result["num_cities"],
        budget=q_result["budget"],
        total_distance=q_result["total_distance"],
        prize_collected=q_result["prize_collected"],
        runtime_seconds=q_result["runtime_ms"] / 1000,
        route=json.dumps(q_result["route"]),
    )
    p_run = models.AlgorithmRun(
        algorithm=p_result["algorithm"],
        num_cities=p_result["num_cities"],
        budget=p_result["budget"],
        total_distance=p_result["total_distance"],
        prize_collected=p_result["prize_collected"],
        runtime_seconds=p_result["runtime_ms"] / 1000,
        route=json.dumps(p_result["route"]),
    )

    db.add(q_run)
    db.add(p_run)
    db.commit()
    db.refresh(q_run)
    db.refresh(p_run)

    # Determine winners
    winner_by_prize = "q-learning" if q_result["prize_collected"] >= p_result["prize_collected"] else "p-marl"
    winner_by_runtime = "q-learning" if q_result["runtime_ms"] <= p_result["runtime_ms"] else "p-marl"

    return SolveResponse(
        q_learning=AlgorithmResult(**q_result),
        p_marl=AlgorithmResult(**p_result),
        winner_by_prize=winner_by_prize,
        winner_by_runtime=winner_by_runtime,
        q_learning_run_id=q_run.id,
        p_marl_run_id=p_run.id,
    )


@app.get("/runs", response_model=list[RunSummary])
def list_runs(db: Session = Depends(get_db)):
    """Return all stored algorithm runs, most recent first."""
    runs = db.query(models.AlgorithmRun).order_by(models.AlgorithmRun.created_at.desc()).all()
    return runs


@app.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific algorithm run by ID."""
    run = db.query(models.AlgorithmRun).filter(models.AlgorithmRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run