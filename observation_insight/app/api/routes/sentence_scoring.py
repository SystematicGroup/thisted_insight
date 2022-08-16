from fastapi import APIRouter
from starlette.requests import Request
from fastapi import Response, status

from observation_insight.app.models.thisted_state import ThistedState
from observation_insight.app.models.sentence_scoring import SentenceScoringPayload


router = APIRouter()

@router.post("/sentencescoring", name="Logging the final user chosen colorings to the backend")
def final_log(
    request: Request,
    data: SentenceScoringPayload 
):
    """
    # Logging final user choices with user choices with username 
    # and which sentences user says belong to which observations.

    ## Params
    * username: str
    * text: str
    * observations: [str]
    * sentences: [{'sentence':str , 'observations':[str]}]

    ## Example
    * 

    ## Returns
    * Writing the input dict to the logfile
    """

    model: ThistedState = request.app.state.model
    model.logfile.write_log("api/sentencescoring", data.json())
    return Response(status_code=status.HTTP_200_OK)