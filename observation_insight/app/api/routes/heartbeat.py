
from fastapi import APIRouter

from observation_insight.app.models.heartbeat import HearbeatResult

router = APIRouter()


@router.get("/heartbeat", response_model=HearbeatResult, name="heartbeat")
def get_hearbeat() -> HearbeatResult:
    """
    Heartbeat monitoring (ensure web client and server connection)
    """
    heartbeat = HearbeatResult(is_alive=True)
    return heartbeat