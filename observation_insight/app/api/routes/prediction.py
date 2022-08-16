from fastapi import APIRouter, Depends
from starlette.requests import Request

from observation_insight.app.models.thisted_state import ThistedState
from observation_insight.app.models.payload import PredictionPayload
from observation_insight.app.api.routes import prediction_base

router = APIRouter()

@router.post("/predict", name="Predict Observation Scheme and Probability")
def predict(
    request: Request,
    data: PredictionPayload
):
    """
    # Prediction of observation schemes based on free text documentation

    ## Params
    * text: str

    ## Example
    * "Bo er faldet"

    ## Returns
    * results: dict
        - **result**(list of dicts): name and probability for each obs_scheme
            - **name** (str): Predicted observation scheme
            - **probability** (int): Probability of predicted observation scheme
    """

    model: ThistedState = request.app.state.model
    results = prediction_base.predict(model, data.text)
    model.logfile.write_log("api/predict", results)

    return results