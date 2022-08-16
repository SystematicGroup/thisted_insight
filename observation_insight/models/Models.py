# here, pick and load an already trained model
    # if we want to make predictions on a specific non-hierarchical model, then we just enter its name and the model will be loaded
from socket import SOMAXCONN
import numpy as np
import os
import joblib
import spacy
from observation_insight.data.encode_labels import Encoder
from numpy.core.fromnumeric import argsort
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
from dotenv import load_dotenv
load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR")
ENCODER_DIR = os.getenv("ENCODER_DIR")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress tensorflow messages when loading own pre-trained BERT

class Model():
    def __init__(self, model_dir):
        self.model_dir = model_dir 

    def scheme_predict(self,model, input_txt):
        # Predicting scheme from the flat model
        predicted_scheme = model.predict(input_txt)
        return predicted_scheme

    def schemes_prob(self, model, input_txt):
        # Calculating schemes probabilities from the flat model
        model.enable_categorical = True
        schemes_probs = model.predict_proba(input_txt)       
        return schemes_probs

    def sort_rank(self,input_array):
        # Sorting the schemes based on their probabilities
        ranked_array = np.flip(np.sort(input_array))
        ranked_probs_idx = np.flip(np.argsort(input_array))
        return ranked_array, ranked_probs_idx
    
    def get_topK(self,top_k, sorted_schemes, sorted_probs):
        # Get topK schemes
        # If top_k is set to -1 -> getting all sorted schemes
        if top_k != -1:
            top_schemes = sorted_schemes[:,0:top_k] 
            top_probs = sorted_probs[:,0:top_k]
            return top_schemes, top_probs
        else :
            return sorted_schemes,sorted_probs
        
    def rank_schemes(self, schemes_probs, top_k=-1):  
        # If the model_mode is flat use this function for prediction
        # Just get probs and sort them      
        sorted_ranked_probs, sorted_ranked_schemes = self.sort_rank(schemes_probs)
        topk_schemes , topk_probs = self.get_topK(top_k, sorted_ranked_schemes, sorted_ranked_probs)
        return topk_schemes, topk_probs
    

def load_all_models(args):
    encoder_name = args["encoder_name"]
    encoder = load_encoder(args)   
    nlp = load_tokenizer()
    if encoder_name == 'danishBERT':
        label_encoder = None
        label_encoder_model = None
    else:
        label_encoder,label_encoder_model = load_label_encoder(args)
    classifier = load_classifier(args)
    return encoder, nlp, label_encoder, label_encoder_model, classifier

def load_encoder(args):
    encoder_name = args["encoder_name"]
    if encoder_name == 'LaBSE' :
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('LaBSE')
        return encoder
    
    elif encoder_name == 'roberta-large':
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained(encoder_name)
        encoder = RobertaModel.from_pretrained(encoder_name) 
        return tokenizer, encoder
    
    elif encoder_name == 'laser':
        from laserembeddings import Laser
        encoder = Laser()
        return encoder

    elif encoder_name == 'danishBERT':
        encoder = AutoTokenizer.from_pretrained(f'{ENCODER_DIR}/pytorch_model_bert/tokenizer')
        return encoder

def load_tokenizer():
    nlp = spacy.load("da_core_news_sm")    
    return nlp

def load_label_encoder(args): 
    label_encoder = Encoder(args["encoder_dir"])    
    label_encoder_model = label_encoder.load_encoder(args["encoder_model_name"])
    return label_encoder, label_encoder_model

def load_classifier(args):
    encoder_name = args["encoder_name"]        
    if encoder_name == 'danishBERT':
        classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, from_tf=True)
    else:
        classifier_path = os.path.join(args["model_dir"], args["model_name"])
        classifier = joblib.load(classifier_path)
        
    return classifier