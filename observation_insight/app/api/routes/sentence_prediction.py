from fastapi import APIRouter, Depends
from starlette.requests import Request
import pandas as pd

from observation_insight.app.models.thisted_state import ThistedState
from observation_insight.app.models.payload import SentPredictionPayload, sentpayload_to_dict
from observation_insight.app.core.config import args
from observation_insight.models.SentenceScoring import SentenceScoring

router = APIRouter()

@router.post("/to_sentences", name="Sentence Scoring Prediction")
def predict(
    request: Request,
    data: SentPredictionPayload 
):
    """
    # Prediction of observation schemes based on sentences

    ## Params
    * username: str
    * text: str
    * obstype_chosen: []

    ## Example
    * username: "user"
    * text: "Bo er faldet. Det gør ondt i hans knæ. Han halter på sit højre ben. Kh hjemmehjælper"
    * obstype_chosen: [
            "Psykosocialt (hverdagsobservation)",              
            "Kontakt til læge",                              
            "Funktionsniveau/egenomsorg (hverdagsobservation)" 
        ]

    ## Returns
    * results: dict
        - **result**(list of dicts): sentence and observations
            - **sentence** (str): Sentence from user's text documentation
            - **observation** (list of dicts): List of each observation name and probability for the user-selected obs_schemes
    """

    model: ThistedState = request.app.state.model
    prediction_ = model.sentprediction_
    model.logfile.write_log("api/to_sentences/input", sentpayload_to_dict(data))    

    tokenized_data, obs_results = prediction_.preprocess(data, args['separator_token'])
    
    # Write to log
    model.logfile.write_log("api/to_sentences/prediction", {"username": data.username, "predictions": obs_results})

    # Predict on fine-tuned model
    ss = SentenceScoring()
    finetuned_model = model.sentmodel
    pred_prob = ss.get_predictions(finetuned_model, tokenized_data, checkpoint=args['checkpoint'])

    ypred = pd.DataFrame(pred_prob).rename(columns={0:'probability'})
    ypred['ObservationScheme'] = tokenized_data['ObservationScheme']
    ypred['sentence'] = tokenized_data['sentence']

    prob_thres=[]   
    sentences = prediction_.text_to_sent({"text": data.text})
    for sentence in sentences:
        df_tmp = ypred[ypred['sentence']==str(sentence)]
        df_tmp = df_tmp.drop_duplicates('ObservationScheme', keep='last')
        prob_thres.append(df_tmp)
    
    # Get results
    results = prediction_.output_to_dict(prob_thres)

    # Write to log
    model.logfile.write_log("api/to_sentences/output", {"username": data.username, "results": results})

    return results