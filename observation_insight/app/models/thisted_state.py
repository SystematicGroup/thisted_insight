# Loading all models needed for prediction (encoders, classifier)

from observation_insight.app.services.models import Prediction
from observation_insight.app.core.config import args
from observation_insight.app.models.log_file import LogFile
from observation_insight.app.services.sentence_model import SentPrediction
from observation_insight.models.Models import load_tokenizer
from observation_insight.models.SentenceScoring import SentenceScoring
from observation_insight.data.DataTransformer import DataTransformer
from observation_insight.models.Models import load_tokenizer

class ThistedState():
    def __init__(self):
        self.encoder = None
        self.nlp = load_tokenizer()
        self.label_encoder = None
        self.label_encoder_model = None
        self.clf = None
        self.logfile = LogFile(args['logfile_dir'], args['logfile_name'])

        self.prediction_: Prediction = Prediction(
            encoder_name = args['encoder_name'],
            model_name = args['model_name']
        )

        self.dt = DataTransformer(checkpoint=args['checkpoint'])

        self.sentscoring_= SentenceScoring()
        self.sentprediction_: SentPrediction = SentPrediction(
            self.nlp,
            self.dt,
            args['threshold'],
            thisted_state=self
        )
        self.sentmodel=None

    def load_models(self, args):
        self.encoder, self.label_encoder, self.label_encoder_model, self.clf = self.prediction_.load_all_models(args)
        self.clf.enable_categorical = True

        self.sentmodel = self.sentscoring_.load_model(args['checkpoint_dir'])
        pass