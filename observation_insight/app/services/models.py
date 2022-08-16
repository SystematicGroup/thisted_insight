from observation_insight.app.services.models_BERT import predict_scheme_BERT
from observation_insight.models.predict_model import predict_scheme, get_schemes_names, schemes_to_names
from observation_insight.features.build_features import build_features
from observation_insight.app.core.config import args, BOTTOM_PREDICTION_COUNT, TOP_PREDICTION_COUNT
from observation_insight.models.Models import load_label_encoder, load_classifier, load_encoder
import spacy

class Prediction():

    nlp = spacy.load("da_core_news_sm") 

    def __init__(self, encoder_name, model_name):
        self.encoder = encoder_name
        self.model_name = model_name

    def load_all_models(self, args):
        encoder_name = args["encoder_name"]
        encoder = load_encoder(args)   
        if encoder_name == 'danishBERT':
            label_encoder = None
            label_encoder_model = None
        else:
            label_encoder,label_encoder_model = load_label_encoder(args)
        classifier = load_classifier(args)
        return encoder, label_encoder, label_encoder_model, classifier

    def preprocessing(self, input_text, encoder, nlp):
        data_features = build_features(args, data=input_text, encoder = encoder, nlp = self.nlp)
        return data_features

    def predict_obs_scheme(self, data_features, model, label_encoder, label_encoder_model):
        """
        # Prediction of observation schemes based on free text documentation

        ## Params
        * data_features (embeddings)
        * model (classifier)
        * label_encoder
        * label_encoder_model

        ## Returns:
        - **results** (str): Predicted observation scheme name
        - **predicted_prob_ordered** (float): Probability of predicted observation scheme
        """
        encoder_name = args["encoder_name"]
        if encoder_name == 'danishBERT':
            results, predicted_prob_ordered = predict_scheme_BERT(model,data_features)
            return results, predicted_prob_ordered
        else:
            predicted_schemes_ordered, predicted_prob_ordered = predict_scheme(args, data_features = data_features, classifier = model)
            results = get_schemes_names(args, predicted_schemes_ordered, label_encoder, label_encoder_model)
            return results, predicted_prob_ordered

    def get_results(self, result_scheme, result_prob):
        res = []
        for i in range(len(result_scheme)):
            res.append({"name": result_scheme[i], "probability": float("{:.6f}".format(result_prob[0][i]))})
        
        results = {
            'result': res
            }

        return results

    def medicin_fix(self, results):
        """Letting medicinadministration and medicindispensering appear together

        Args:
            results (dict): Dictionary of prediction results

        Returns:
            results_dict (dict): Dictionary of prediction results with the added functionality
        """
        top_location = TOP_PREDICTION_COUNT-1
        bottom_location = BOTTOM_PREDICTION_COUNT-1
        order = results['result']
        schemes = []
        for element in order:
            schemes.append(element['name'])

        disp_pos = schemes.index('Medicindispensering')
        adm_pos = schemes.index('Medicinadministration ')

        if adm_pos < disp_pos:
            if adm_pos == top_location or adm_pos == bottom_location:
                order.insert(adm_pos-1, order.pop(adm_pos))
                adm_pos = adm_pos-1
                order[adm_pos]['probability'] =  order[adm_pos+1]['probability']+0.000001
            order.insert(adm_pos+1, order.pop(disp_pos))
            order[adm_pos+1]['probability'] = order[adm_pos]['probability']
        elif disp_pos < adm_pos:
            if disp_pos == top_location or disp_pos == bottom_location:
                order.insert(disp_pos-1, order.pop(disp_pos))
                disp_pos = disp_pos-1
                order[disp_pos]['probability'] =  order[disp_pos+1]['probability']+0.000001
            order.insert(disp_pos+1, order.pop(adm_pos))
            order[disp_pos+1]['probability'] = order[disp_pos]['probability']

        results_dict = {'result': order}

        return results_dict