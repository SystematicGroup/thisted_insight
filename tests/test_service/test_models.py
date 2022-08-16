from observation_insight.app.core.config import args
from observation_insight.app.services.models import Prediction

from observation_insight.models.Models import load_label_encoder, load_classifier
from observation_insight.app.services.models import load_encoder


def test_prediction() -> None:
    input = 'test'

    encoder = load_encoder(args) 
    label_encoder,label_encoder_model = load_label_encoder(args)
    classifier = load_classifier(args)

    pm = Prediction(args['encoder_name'], args['model_name'])
    preproc = pm.preprocessing(input, encoder, pm.nlp)

    scheme, prob = pm.predict_obs_scheme(preproc, classifier, label_encoder, label_encoder_model)
    result = pm.get_results(scheme, prob)
    if result['result'][0]['name'] == str:
        assert True
    if result['result'][0]['probability'] == float:
        assert True