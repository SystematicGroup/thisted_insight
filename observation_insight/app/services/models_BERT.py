import torch
import pandas as pd
import numpy as np
from torch.nn.functional import softmax

from observation_insight.models.predict_model import schemes_to_names
from observation_insight.app.core.config import args

def predict_scheme_BERT(model, data_features):
    """Predict the schemes for the input, and the probabilities, sorted by probability

    Args:
        model (classifier): The loaded BERT model
        data_features (list of int): The embedded data as inputID and Attention mask

    Returns:
        predicted_original_names (str): Predicted observation scheme name 
        predicted_prob_ordered (float): Probability of predicted observation scheme
    """
    
    logits = []
    outputs = model(data_features[0], token_type_ids=None, attention_mask=data_features[1])

    logits.append(outputs[0])
    logits = torch.cat(logits, dim=0)
    probs = softmax(logits, dim=1).cpu().detach()[0]

    levels = np.load(f'{args["levels_dir"]}/levels.npy', allow_pickle=True)
    observationScheme = []
    for i in range(len(probs)):
        observationScheme.append(levels[i])

    df = pd.DataFrame(data={'observationScheme': observationScheme, 'probability': probs})
    df.sort_values('probability', ascending=False, inplace=True)

    predicted_original_names = schemes_to_names(df['observationScheme'].to_list(), args["encoder_dir"], args["converter_name"])
    predicted_prob_ordered = [df['probability'].to_list()]

    return predicted_original_names, predicted_prob_ordered