from typing import List
from starlette.testclient import TestClient
import pandas as pd
from observation_insight.app.main import app
from observation_insight.app.models.thisted_state import ThistedState
from observation_insight.app.core.config import args
from observation_insight.models.SentenceScoring import SentenceScoring
from observation_insight.app.models.payload import SentPredictionPayload

client = TestClient(app)
@app.get("/to_sentences")

def test_prediction_sent():
    input = {
        "username": "",
        "text": "Jeg skriver noget meget seriøs dokumentation. Bo er faldet. Dragons are here. Det gør ondt i hans knæ.",
        "obstype_chosen": [
            "Psykosocialt (hverdagsobservation)",              
            "Kontakt til læge",                                
            "Funktionsniveau/egenomsorg (hverdagsobservation)" 
        ]
    }

    input = SentPredictionPayload(
        username=input['username'],
        text=input['text'],
        obstype_chosen=input["obstype_chosen"]
    )

    thisted_state = ThistedState()
    thisted_state.load_models(args)

    sp = thisted_state.sentprediction_
    tokenized_data, _ = sp.preprocess(input, args['separator_token'])
    
    # Predict on fine-tuned model
    ss = SentenceScoring()
    finetuned_model = ss.load_model(args["checkpoint"])
    pred_prob = ss.get_predictions(finetuned_model, tokenized_data,checkpoint=args['checkpoint'])

    ypred = pd.DataFrame(pred_prob).rename(columns={0:'probability'})
    ypred['ObservationScheme'] = tokenized_data['ObservationScheme']
    ypred['sentence'] = tokenized_data['sentence']
    ypred['text'] = tokenized_data['text'] 

    prob_thres=[]
    sentences = sp.text_to_sent({"text": input.text})
    for sentence in sentences:
        df_tmp = ypred[ypred['sentence']==str(sentence)]
        df_tmp = df_tmp.drop_duplicates('ObservationScheme', keep='last')
        prob_thres.append(df_tmp)
    
    # Get results
    results = sp.output_to_dict(prob_thres)

    if results['observations'][0] == dict: assert True
    if results['observations'][0]['name'] == str: assert True
    if results['observations'][0]['description'] == str: assert True
    if results['observations'][0]['conditions'] == str: assert True

    if results['result'][0]['sentence'] == str: assert True
    if results['result'][0]['observations'] == List: assert True
    if results['result'][0]['observations'][0] == dict: assert True
    if results['result'][0]['observations'][0]['ObservationScheme'] == str: assert True
    if results['result'][0]['observations'][0]['probability'] == float: assert True