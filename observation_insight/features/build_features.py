import pandas as pd
from observation_insight.data.clean_data import get_text_data, tokenizing
from observation_insight.features.embeddings import text_embedder

def build_features(args, data, encoder, nlp):
    # If the input data is loaded from a file, get text column  ---> input_flag = True 
    input_flag = isinstance(data, pd.DataFrame)
    if input_flag:
        data = get_text_data(data, args["col_txt_name"])

    # Making embeddings
    encoder_name = args["encoder_name"]

    # Tokenizing and embedding the input text. When using danishBERT the data is not tokenized before embedding
    if encoder_name == 'danishBERT':
        data_features = text_embedder(args, data, encoder, input_flag)
    else:
        data_tokenized = tokenizing(data, nlp, input_flag)
        data_features = text_embedder(args, data_tokenized, encoder, input_flag)
    return data_features