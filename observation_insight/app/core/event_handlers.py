# Event-handlers: functions that need to be executed before the application starts up,
# or when the application is shutting down

# Here, we load all the models defined in ThistedState when starting up the application
# and returning to None state when shutting down the application

from typing import Callable

from fastapi import FastAPI
from loguru import logger
from observation_insight.app.models.thisted_state import ThistedState
from observation_insight.app.core.config import args

def _startup_model(app: FastAPI) -> None:
    """
    Initiate models (encoders, classifiers)
    """
    model_instance = ThistedState()
    app.state.model = model_instance
    # comment next line if you don't need predictions and need fast startup (for testing)
    app.state.model.load_models(args)

def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None

def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)
    return startup

def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)
    return shutdown