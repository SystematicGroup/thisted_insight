from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

class Encoder():
    def __init__(self, encoder_dir):
        self.encoder_dir = encoder_dir 
        
    def load_encoder(self, encoder_name):    
        # Loading label encoder
        model_path = os.path.join(self.encoder_dir, encoder_name)
        encoder = LabelEncoder()
        encoder.classes_ = np.load(model_path, allow_pickle = True)    
        return encoder

    def inverse_transform(self, encoder, predicted_schemes):
        # Transforming scheme values to their names
        predicted_schemes_list = predicted_schemes.reshape(max(predicted_schemes.shape), -1).tolist()
        decoded_schemes = encoder.inverse_transform(predicted_schemes_list)
        return decoded_schemes