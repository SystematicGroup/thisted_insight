# Load a model and make prediction on new data
import pandas as pd
import os 
from observation_insight.models.Models import Model
import datetime
import dateutil.tz


def predict_scheme(args, data_features, classifier):
    # Predicting schemes based on the model type and rank them
    model = Model(args["model_dir"])    
    model.enable_categorical = True
    schemes = model.schemes_prob(classifier, data_features)
    ranked_schemes, ranked_probs = model.rank_schemes(schemes)
    return ranked_schemes, ranked_probs
    
def read_csv_file(file_dir, file_name):
    file_path = os.path.join(file_dir, file_name)
    converter_pattern = pd.read_csv(file_path)
    return converter_pattern

def schemes_to_names(decoded_schemes,converter_path , converter_name):
    # Loading the file contains converter names
    # Retieving the original nmaes from the file
    converter_pattern = read_csv_file(converter_path, converter_name)
    decoded_schemes = pd.DataFrame(decoded_schemes, columns=['clean_scheme'])    
    merged_cleaned_schemes = decoded_schemes.merge(converter_pattern, on=['clean_scheme'], sort = False)
    predicted_schemes_names = merged_cleaned_schemes ['original_scheme']
    return predicted_schemes_names

def save_pred_schemes(args, results):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    out_name = 'predictions_%s_%s'  % (timestamp,args["data_name"])
    out_path = os.path.join(args["output_dir"], out_name)
    results.to_csv(out_path, index=False)

def get_schemes_names(args, predicted_schemes_ordered, encoder, encoder_model):    
    # Inversing scheme values to their names
    # Converting names to the original names
    decoded_schemes = encoder.inverse_transform(encoder_model, predicted_schemes_ordered)
    predicted_original_names = schemes_to_names(decoded_schemes, args["encoder_dir"], args["converter_name"])
    return predicted_original_names