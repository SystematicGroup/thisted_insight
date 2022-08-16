from fastapi import FastAPI
from fastapi.testclient import TestClient
from observation_insight.app.services.models import Prediction
from observation_insight.app.core.config import args

app=FastAPI()

@app.get("/")
def read_input():
    input = 'Bo er faldet og har ondt i knæet'
    p = Prediction(args['encoder_name'], args['model_name'])
    encoder, label_encoder, label_encoder_model, clf = p.load_all_models(args)

    data_features = p.preprocessing(input, encoder, nlp=None)
    result_scheme, result_prob = p.predict_obs_scheme(data_features, clf, label_encoder, label_encoder_model)
    results = p.get_results(result_scheme, result_prob)

    return results['result'][0]

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200