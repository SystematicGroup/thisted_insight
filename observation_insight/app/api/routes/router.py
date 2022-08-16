# Routing API requests for the api/routes/prediction.py
    # Determines which actions that can be executed

from fastapi import APIRouter
from observation_insight.app.api.routes import heartbeat, prediction, sentence_prediction, sentence_scoring, obslist

api_router = APIRouter()
api_router.include_router(prediction.router, tags=["Observation Classification"])
api_router.include_router(heartbeat.router, tags=["health"], prefix="/health")
api_router.include_router(sentence_prediction.router, tags=['Sentence Scoring'])
api_router.include_router(sentence_scoring.router, tags=['Final Sentence Scoring Logging'])
api_router.include_router(obslist.router, tags=['List of observations'])