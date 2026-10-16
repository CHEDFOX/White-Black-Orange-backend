"""Version 1 router."""

from fastapi import APIRouter

from app.api.v1 import analysis, forecasts, instruments, market_data, reports, system, watchlists

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(instruments.router)
api_router.include_router(market_data.router)
api_router.include_router(analysis.router)
api_router.include_router(forecasts.router)
api_router.include_router(reports.router)
api_router.include_router(watchlists.router)
