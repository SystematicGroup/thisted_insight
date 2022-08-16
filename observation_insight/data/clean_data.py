import os
import pandas as pd
from stop_words import get_stop_words

        
def read_data(args):
    data_path = os.path.join(args["data_dir"], args["data_name"])
    data = pd.read_excel(data_path)
    return data 

def get_text_data(text, col_txt_name ):
    text_data = text[col_txt_name]
    return text_data

def tokenizing(text,nlp,input_flag):
    
    da_stopwords = get_stop_words('da')

    # If the input data is dataframe -> loaded from the file
    if input_flag == True: 
        # Data tokenization by spacy
        text = text.apply(nlp) 
        # Remove numbers, remove stop words, convert all text to lower case
        text = text.apply(lambda x: [item.text.lower() for item in x if (item.text not in da_stopwords and item.text.isalpha()== True)])

    # If the input text is just a string
    else:
        text = nlp(text) 
        # Create list of word tokens
        token_list = []
        for token in text:
            token_list.append(token.text)
        text = [item.lower() for item in token_list if (item not in da_stopwords and item.isalpha()== True)]
    
    return text