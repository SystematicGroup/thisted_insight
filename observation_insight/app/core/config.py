import os
from starlette.config import Config
from dotenv import load_dotenv
load_dotenv()

APP_VERSION = "0.0.1"
APP_NAME = "CURA - Observation Scheme Prediction"
API_PREFIX = "/api"

ENCODER_DIR = os.getenv("ENCODER_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
LOGFILE_DIR = os.getenv("LOGFILE_DIR")
SEPARATOR_DIR = os.getenv("SEPARATOR_DIR")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR")
MAUS_FILE_DIR = os.getenv("MAUS_FILE_DIR")
LEVELS_DIR = os.getenv("LEVELS_DIR")
OBSLIST_DIR = os.getenv("OBSLIST_DIR")
DISABLE_CUDA = os.getenv("DISABLE_CUDA").lower() in ('true')
DESCRIPTION_AND_CONDITION_DIR = os.getenv("DESCRIPTION_AND_CONDITION_DIR")
TOP_PREDICTION_COUNT = int(os.getenv("TOP_PREDICTION_COUNT"))
BOTTOM_PREDICTION_COUNT = int(os.getenv("BOTTOM_PREDICTION_COUNT"))

args = {
    "encoder_name": "danishBERT", #'LaBSE',
    "encoder_dir": ENCODER_DIR, 
    "model_dir": MODEL_DIR, 
    "model_name": "pytorch_model.bin",
    "encoder_model_name": 'classes_subsample_100000.npy',
    "converter_name": 'convert_schemes_pattern.csv',
    "model_mode": "flat",
    "logfile_dir":LOGFILE_DIR,
    "logfile_name": 'results_logs.txt',
    "levels_dir":LEVELS_DIR,
    "disable_cuda":DISABLE_CUDA,

    # Sentence scoring
    'separator_token':'SEP', #'CLS_SEP','COMMA'
    'threshold':0.048,
    'checkpoint': "Maltehb/aelaectra-danish-electra-small-cased",
    'checkpoint_dir': CHECKPOINT_DIR,
    'num_labels':2,
    'separator_data_dir': SEPARATOR_DIR,
    'description_and_condition_dir': DESCRIPTION_AND_CONDITION_DIR,
    'obslist_dir':OBSLIST_DIR
}

config = Config(".env")
IS_DEBUG: bool = config("IS_DEBUG", cast=bool, default=False)
