import os
from observation_insight.app.core.config import args
if args['disable_cuda']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from observation_insight.app.api.routes.router import api_router
from observation_insight.app.core.config import (API_PREFIX, APP_NAME, APP_VERSION, IS_DEBUG)
from observation_insight.app.core.event_handlers import (start_app_handler,
                                                  stop_app_handler)
import observation_insight.app.core.config as cfg

maus_file_location = cfg.MAUS_FILE_DIR

# Origins: the frontend running in a browser has JavaScript code that communicates
# with backend, and the backend is in a different "origin" than the frontend.
origins = [
    "http://localhost"
]

def get_app() -> FastAPI:    
    fast_app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=IS_DEBUG)
    fast_app.include_router(api_router, prefix=API_PREFIX)

    fast_app.add_event_handler("startup", start_app_handler(fast_app))
    fast_app.add_event_handler("shutdown", stop_app_handler(fast_app))

    fast_app.add_middleware(
        CORSMiddleware, 
        allow_origins=origins,  # A list of origins that should be permitted to make cross-origin requests
        allow_credentials=True, # Indicate that cookies should be supported for cross-origin requests.
        allow_methods=["*"],    # All standard methods (e.g. GET and POST)
        allow_headers=["*"],    # All headers
    )

    fast_app.mount("/", StaticFiles(directory=maus_file_location, html=True), name="static maus handler")
    # "Mounting" means adding a completely "independent" application in a specific path, that then takes care
    # of handling everything under that path, with the path operations declared in that sub-application.

    return fast_app

app = get_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)