import numpy as np
import torch
from observation_insight.app.core.config import args

def text_embedder (args, data_tokenized, encoder, input_flag):
    encoder_name = args["encoder_name"]
    if encoder_name == 'LaBSE' :  
        if  input_flag:    
            data_encoded = data_tokenized.apply(lambda x: encoder.encode([(' '.join([str(elem) for elem in x]))]))
        else:
            data_encoded = [encoder.encode([(' '.join([str(elem) for elem in data_tokenized]))])]
        data_encoded = np.vstack(data_encoded)   
        return data_encoded
    
    elif encoder_name == 'roberta-large':        
        data_encoded = data_tokenized.apply (lambda x: torch.mean(encoder(**tokenizer([' '.join([str(elem) for elem in x])], return_tensors= "pt", max_length=512)).last_hidden_state [0] , 0).detach().numpy())
        data_encoded = np.vstack(data_encoded)
        return data_encoded
    
    elif encoder_name == 'laser':        
        data_encoded = encoder.embed_sentences(data_tokenized, lang='da') 
        return data_encoded

    elif encoder_name == 'danishBERT':
        # When using danishBERT, data_tokenized is raw text, and have not been tokenized beforehand
        bert_inp=encoder.encode_plus(data_tokenized,add_special_tokens = True,max_length = 128, padding = 'max_length',return_attention_mask = True, truncation = True, return_tensors = 'pt')
        data_encoded = [torch.cat([bert_inp['input_ids']], dim=0), torch.cat([bert_inp['attention_mask']], dim=0)]
        return data_encoded