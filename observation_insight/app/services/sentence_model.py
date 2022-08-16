# MAKE DATA PRE-PROCESSING FOR INFERENCE (SENTENCE SCORING)
from logging import raiseExceptions
import json
import json
import spacy
import pandas as pd

pd.set_option('display.max_colwidth', None)
import pyarrow as pa
from datasets import Dataset
import os

from observation_insight.data.DataTransformer import DataTransformer 
from observation_insight.models.SentenceScoring import SentenceScoring
from observation_insight.app.core.config import args
from observation_insight.app.models.payload import SentPredictionPayload, sentpayload_to_dict
from observation_insight.models.Models import load_tokenizer
from observation_insight.app.api.routes import prediction_base
import observation_insight.app.core.config as cfg
LOGFILE_DIR = os.getenv("LOGFILE_DIR")
SEPARATOR_DIR = os.getenv("SEPARATOR_DIR")


def load_sentmodels():
    nlp = load_tokenizer()
    scheme_question_bank = None 
    return nlp, scheme_question_bank


class ObsSchemesQuestionMap:
    '''
    This class is made to make a mapping from ObservationScheme to the list of related ObservationQuestions
   
    '''
    def __init__(self):
        self.record = {} 

    def __getitem__(self,key):
        # Retrieves the list of questions from a specific ObservationScheme
        return self.record[key]

    def __setitem__(self, key, value):
        # Adds an attributes to the class
        # Each attribute is an ObservationScheme with a list of related questions
        self.record[key] = value

def read_csv_file (file_dir, file_name, sep=''):
    file_path = os.path.join(file_dir, file_name)
    if sep == '':
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path, sep=sep)
    return data

def make_obsScheme_dict (data_dir , data_name, obs_col, q_col):
    '''
        This function makes a mapping from ObservationScheme to the related ObservationQuestions

        inputs
        ---------------------
        data_dir: The dir to the original csv file
        data_name: The name of the data file 
        obs_col: The name of the column contains the ObservationSchme
        q_col: The name of the column contains ObservationQuestion
        
        outputs
        ---------------------
        A bank mapper which maps ObservationScheme to the realted ObservationQuestion
    '''
    # Make an instance of class mapper
    schemes_questions_bank = ObsSchemesQuestionMap()

    # Loading original data
    data = read_csv_file(data_dir, data_name)

    # Get unique obs_schemes names 
    obs_schemes = data[obs_col].unique()
    for obs in obs_schemes:
        # For each scheme make a list of questions
        # Add an attribute for each unique ObservationScheme
        questions = data[data[obs_col]== obs][q_col].unique()
        schemes_questions_bank[obs] = questions   

    return schemes_questions_bank


class SentPrediction():
    def __init__(self, nlp, dt, threshold, thisted_state):
        """
        Constructs all necessary attributes for preprocessing the input text for sentence scoring.

        Input
        ---------
            nlp: spacy
                Loaded spacy "da_core_news_sm" for sentence segmenter
            dt: object
                An instance of the DataTransformer object, incl. a 'checkpoint' (model)
            threshold: float
                A flot value between 0-1 for the cutoff to when we will add the model's suggestion
            scheme_question_bank: dict
                Generate a class of observations and their questions (key: obs_scheme, value: list of questions)
            thisted_state

        Returns
        ---------
            data: Preprocessed data ready for tokenization
        """
        self.nlp = nlp
        self.dt = dt
        self.threshold = threshold
        self.thisted_state = thisted_state

    def _add_obsschme_from_model(self, input, log_classification):
        """
        Adding obs_schemes in the input dict from the application (v2) that includes
        the obs_scheme (with highest prob) chosen by the model.
        
        Input
        ---------
            input: SentPredictionPayload
                The input SentPredictionPayload received by MAUS with the user's name, text and chosen obs_schemes
            log_classification: dict
                A log file from the users free text documentation

        Returns
        ---------
            p_input: dict
                An updated dictionary including observation scheme adde by the model
        """
        p_input = sentpayload_to_dict(input)
        p_input['obstype_all'] = p_input['obstype_chosen']   

        return p_input
      
    def text_to_sent(self, input_tmp):
        """
        Input
        ---------
            input: dict
                The updated input incl. observations selected by user and model
                Dictionary needs to have at least one key named 'text'

        Returns
        ---------
            sentences: list
                A list of sentences from the input text
        """
        if isinstance(input_tmp, SentPredictionPayload):
            raise Exception("Expected dictionary but received SentPredictionPayload")

        doc = self.nlp(input_tmp["text"])
        sentences = [sent for sent in doc.sents]
        return sentences

    def _concatenate_separator_token(self, input, sentences, separator_token):
        """
        Add new column where we concatenate scheme, question and text with a separator token.

        Input
        --------
            df_exp: pd.DataFrame
                Dataframe with ObservationScheme and ObservationQuestion columns
            sentences: list
                list of sentences from the input
            separator_token: str
                Either 'COMMA', 'SEP' or 'CLS_SEP'
        
        Returns
        --------
            obs_q_ans: list of pd.DataFrames
                List of dataframes, each including 'text'column with scheme, question and input text in separators,
                observation scheme-column and related question-column.
        """
        if separator_token == "COMMA":
            split_token_list = ['  ,','  ,']
        if separator_token == "SEP":
            split_token_list  = ['', ' [SEP] ']
        if separator_token == "CLS_SEP":
            split_token_list  = ['[CLS] ', ' [SEP] ']

        df_obsanswer = pd.DataFrame(input).rename(columns={'text':'sentence', 'obstype_all':'ObservationScheme'})
        df_obsanswer['text'] = split_token_list[0] + df_obsanswer['ObservationScheme'] + split_token_list[1]

        df_s = []
        for s in range(len(sentences)):
            df_obsanswer2 = df_obsanswer.copy()
            df_obsanswer2["text"] = [f'{i}{sentences[s]}' for idx, i in enumerate(df_obsanswer.text)]
            df_obsanswer2["sentence"] = [f'{sentences[s]}' for idx, i in enumerate(df_obsanswer2.index)]
            df_s.append(df_obsanswer2)

        df_s2= pd.concat(df_s)

        return df_s2[['text','ObservationScheme','sentence']]

    def _to_DatasetType(self, df):
        return Dataset(pa.Table.from_pandas(df))

    def preprocess(self, input, separator_token):
        """
        Preprocess the input received by the Insights demo-application (v2)

        Input
        --------
            input: SentPredictionPayload
                The input SentPredictionPayload received by MAUS with the user's name, text and chosen obs_schemes
            separator_token: str
                Either 'COMMA', 'SEP' or 'CLS_SEP'

        Returns
        --------
            data: DatasetType
                Text splitted into sentences and concatenated with separator token for each of the chosen obs_schemes and their related questions
        """

        obs_results = prediction_base.predict(self.thisted_state, input.text)

        p_input:dict = self._add_obsschme_from_model(input, obs_results)
        sentences = self.text_to_sent(p_input)

        df_obsanswer = self._concatenate_separator_token(p_input, sentences, separator_token)
        
        data = self._to_DatasetType(df_obsanswer)
        tokenized_data = self.dt.tokenize(data)
        return tokenized_data, obs_results

    def _get_functional_condition(self, scheme:str, dir: str):
        df = pd.read_csv(f'{dir}/ObservationConditions_formatted.csv', sep=';')
        df.set_index('ObservationScheme', inplace=True)
        condition = ""
        try:
            condition = df.loc[scheme,'ConditionName']
        except:
            pass
        return condition
    
    def _get_descriptions(self, scheme:str, dir: str):
        df = pd.read_csv(f'{dir}/observationtypes_description.csv')
        df.set_index('obsscheme', inplace=True)
        description = ""
        try:
            description = df.loc[scheme,'description']
        except:
            pass
        return description

    def output_to_dict(self,prob_thres):
        result_dict = {
            'observations':[],
            'result':[]
        }
        
        # Observations
        schemes = prob_thres[0].groupby(['ObservationScheme'])
        for scheme, gp in schemes:
            gp_dict = {}
            gp_dict['name'] = scheme
            gp_dict['description'] = self._get_descriptions(scheme, dir=cfg.DESCRIPTION_AND_CONDITION_DIR)
            gp_dict['conditions'] = self._get_functional_condition(scheme, dir=cfg.DESCRIPTION_AND_CONDITION_DIR)
            result_dict['observations'].append(gp_dict)

        # Results
        for i in range(len(prob_thres)):
            groups = prob_thres[i].groupby(['sentence'])
            for k, gp in groups:
                gp_dict = {}
                gp_dict['sentence'] = k + ' ' # add space after each sentence
                gp_dict['observations'] = gp [['ObservationScheme','probability']].to_dict(orient='record')
                result_dict['result'].append(gp_dict)
        
        return result_dict


if __name__=="__main__":
    nlp = spacy.load("da_core_news_sm")
    dt = DataTransformer(checkpoint=args["checkpoint"])

    input = {
        "username": "user",
        "text": "Jeg skriver noget meget seriøs dokumentation. Bo er faldet. Det gør ondt i hans knæ. Kh",
        "obstype_chosen": [
            "Psykosocialt (hverdagsobservation)",              # obs1
            "Kontakt til læge",                                # obs2
            "Funktionsniveau/egenomsorg (hverdagsobservation)" # obs4
        ]
    }


    # Obsscheme_question "bank"
    dir = '/interim/Sentence_scoring/superuserdata/sep_by_CLS_comma_SEP'        
    name = 'all_combined_fields_by_CLS_comma_SEP_pos_neg_data_and_comments_superusers.csv'

    scheme_question_bank = make_obsScheme_dict(
        data_dir=dir,
        data_name=name,
        obs_col='ObservationScheme',
        q_col='ObservationQuestion'    
    )

    log_name = "log_classification_prediction_date"


    # Concatenate with separator token
    separator_token = args['separator_token']

    # Preprocess input
    ppi = SentPrediction(nlp, dt, 0.048) 
    tokenized_data, _ = ppi.preprocess(input=input, separator_token=separator_token)


    # Predict on fine-tuned model
    checkpoint_new = '/interim/Sentence_scoring/fine_tuned_model'
    ss = SentenceScoring()
    finetuned_model = ss.load_model(checkpoint_new)
    pred_prob = ss.get_predictions(finetuned_model, tokenized_data,checkpoint=args['checkpoint'])

    prob_thres=[]
    ypred = pd.DataFrame(pred_prob).rename(columns={0:'probability'})
    ypred['ObservationScheme'] = tokenized_data['ObservationScheme']
    ypred['ObservationQuestion'] = tokenized_data['ObservationQuestion']
    ypred['sentence'] = [tokenized_data['Sentence'][i][0] for i in range(len(tokenized_data['Sentence']))]
    
    sentences = ppi.text_to_sent(input)
    for sentence in sentences:
        df_tmp = ypred[ypred['sentence']==str(sentence)]
        df_tmp = df_tmp.drop_duplicates('ObservationScheme', keep='last')
        prob_thres.append(df_tmp)

    # To dict
    result_dict = ppi.output_to_dict(prob_thres)
    with open('json_result.json', 'w') as outfile:
        json.dump(result_dict, outfile)