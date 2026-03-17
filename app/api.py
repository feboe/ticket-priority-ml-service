"""FastAPI application for the ticket triage demo."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from app.demo_tickets import DEMO_TICKETS, get_demo_ticket
from app.service import TicketRoutingService, get_default_service


class TicketRequest(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class TaskPredictionResponse(BaseModel):
    label: str
    runner_up_label: str
    margin_gap: float


class TaskModelMetadataResponse(BaseModel):
    run_id: str
    run_name: str
    algorithm: str
    model_family: str
    analyzer: str
    feature_families: list[str]


class PredictResponse(BaseModel):
    input: dict[str, str]
    predictions: dict[str, TaskPredictionResponse]
    models: dict[str, TaskModelMetadataResponse]


class HealthResponse(BaseModel):
    status: str
    tasks: list[str]
    models: dict[str, TaskModelMetadataResponse]


class DemoTicketResponse(BaseModel):
    index: int
    total: int
    title: str
    ticket: dict[str, str]


def create_app(service: TicketRoutingService | None = None) -> FastAPI:
    resolved_service = service or get_default_service()
    app = FastAPI(title=resolved_service.title, version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        return resolved_service.health()

    @app.get("/demo-ticket", response_model=DemoTicketResponse)
    def demo_ticket(index: int = Query(default=0, ge=0)) -> dict[str, Any]:
        resolved_index, ticket = get_demo_ticket(index)
        title = ticket.pop("title")
        return {
            "index": resolved_index,
            "total": len(DEMO_TICKETS),
            "title": title,
            "ticket": ticket,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: TicketRequest) -> dict[str, Any]:
        return resolved_service.predict_ticket(
            subject=request.subject,
            body=request.body,
        )

    return app


app = create_app()
