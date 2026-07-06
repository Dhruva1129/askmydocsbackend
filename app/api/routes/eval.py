import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.models.models import EvalRun
from app.schemas.schemas import EvalRunOut
from app.core.config import settings

router = APIRouter()


@router.get("/history", response_model=list[EvalRunOut])
async def eval_history(db: AsyncSession = Depends(get_db)):
    """Return last 50 eval runs for the dashboard."""
    result = await db.execute(
        select(EvalRun).order_by(EvalRun.created_at.desc()).limit(50)
    )
    return result.scalars().all()


@router.get("/latest", response_model=EvalRunOut)
async def latest_eval(db: AsyncSession = Depends(get_db)):
    """Return the most recent eval run."""
    result = await db.execute(
        select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "No eval runs found. Run: pytest tests/")
    return run


@router.get("/thresholds")
async def get_thresholds():
    """Expose configured quality thresholds for the frontend dashboard."""
    return {
        "faithfulness": settings.MIN_FAITHFULNESS,
        "answer_relevancy": settings.MIN_ANSWER_RELEVANCY,
        "context_recall": settings.MIN_CONTEXT_RECALL,
    }
