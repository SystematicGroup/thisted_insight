# Thisted live demo app - thisted_insight

This a demo implementation of the Thisted project functionality. 

The following components are used and included:
* Cookie-cutter for Data Science <https://drivendata.github.io/cookiecutter-data-science/>
* FastAPI [Sebastián Ramírez](https://github.com/tiangolo)
* FastAPI folder structure [cookiecutter-fastapi](https://github.com/arthurhenrique/cookiecutter-fastapi)
* Notebooks

## Requirements
Python 3.8+

## Installation
Install the required packages in your local environment
```bash
conda create --name thisted_observation_insight python=3.8
conda activate thisted_observation_insight
pip install -r requirements.txt
``` 

## Setup
1. Duplicate the `.env.example` file and rename it to `.env` 

2. In the `.env` file configure the entries found in the file to the correct settings<br>

You need the following data and model files

1. LaBSE_XGBOOST_Sub.100000_embd_768_LR.0.010_gamma.0.010_MaxEpoch.1000_stop.1000.sav

2. all_combined_fields_by_CLS_comma_SEP_pos_neg_data_and_comments_superusers.csv

3. classes_subsample_100000.npy

4. convert_schemes_pattern.csv

5. joblib_LaBSE_XGBOOST_Sub.100000_embd_768_LR.0.010_gamma.0.010_MaxEpoch.1000_stop.1000.sav

6. config.json

7. pytorch_model.bin

8. training_args.bin

9. Observation_types.csv

10. levels.npy

11. observationtypes_description.csv

These can be found on Move-IT


## Run API locally

1. Start your  app with: 

```bash
uvicorn observation_insight.app.main:app
```

or for debugging you can run with

```bash
python observation_insight/app/main.py
```

2. Go to [http://localhost:8000/docs](http://localhost:8000/docs).
   
4. You can use the sample payload from the `tests/test_service/test_models.py`.

## Run API with Docker
1. Edit `Dockerfile` if necessary
2. Build the docker image

```bash
docker build -t project_template .
```

3. Deploy the image in a container
```bash
docker run -d --name project_template -p 8000:80 project_template
```

## Test in Notebook
1. Run `jupyter lab` in a terminal
2. Open the notebook in `notebooks/api_test.ipynb`

## Tests
1. Please navigate to the `tests` folder in the terminal
2. Run `pytest` in the terminal to showcase automatic tests

## Folder structure

```
├── observation_insight     <- Source code for use in this project. 
│   ├── app                                 <- FastAPI web related stuff.  
│   │   ├── api                             <- API stuff.
│   │   │   └── routes                      <- web routes.
│   │   │       ├── heartbeat.py            <- Heartbeat monitoring
│   │   │       ├── obslist.py              <- List of possible observation types
│   │   │       ├── prediction_base.py      <- Make prediction, with pre- and postprocessing
│   │   │       ├── prediction.py           <- Prediction route. What should happen, if request from prediction route is sent?
│   │   │       ├── router.py               <- Routing API requests
│   │   │       ├── sentence_prediction.py  <- Make sentence scoring prediction
│   │   │       └── sentence_scoring.py     <- Log final choice
│   │   ├── core                            <- application configuration, startup events, logging.      
│   │   │   ├── config.py                   <- Configurations: arguments, app version, project root
│   │   │   ├── event_handlers.py           <- Functions that need to be executed before application starts up/shuts down
│   │   │   ├── messages.py                 <- Messages, e.g. if no API key is needed
│   │   │   └── secutiry.py                 <- Validate requests
│   │   ├── models                          <- pydantic models for this application.
│   │   │   ├── heartbeat.py                <- Is web client and server connection alive (true/false)
│   │   │   ├── log_file.py                 <- Logging to file
│   │   │   ├── payload.py                  <- The body of the request. Contains the data that is send to the server when making an API request.
│   │   │   ├── prediction.py               <- Base model for classification
│   │   │   ├── sentence_scoring.py         <- Base model for sentence scoring
│   │   │   └── thisted_state.py            <- Class for loading all necessary models, e.g. encoder and classifier
│   │   ├── services                        <- wrapper to call DS stuff (the services we provide)
│   │   │   ├── models_BERT.py              <- Predict classification using BERT
│   │   │   ├── models.py                   <- Functions and classes for load making a inference (preprocessing data, prediction)
│   │   │   └── sentence_model.py           <- Predict sentence scoring and preprocessing
│   │   ├── main.py                         <- FastAPI application with name, version, router and event handlers
│   │   └── README_API.md                   <- ReadMe specific for API
│   │
│   ├── data                <- Scripts to download or generate data
│   │   ├── clean_data.py                   <- Clean input before inference
│   │   ├── DataTransformer.py              <- Create sentence scoring object
│   │   └── encode_labels.py                <- Encode labels and transform scheme values to names
│   │
│   ├── features            <- Scripts to turn raw data into features for modeling
│   │   ├── build_features.py               <- Build features, tokenize and embed data.
│   │   └── embeddings.py                   <- Embedders for differnet embedding algorithms
│   │
│   ├── models              <- Scripts to train models and then use trained models to make predictions         
│   │   ├── Models.py                       <- Loading model functions
│   │   ├── predict_model.py                <- Predict model functions
│   └── └── SentenceScoring.py              <- Sentence scoring functions
│
├── tests                   <- Pytest test
│   │   ├── test_api
│   │   │   ├── test_heartbeat.py           <- API heartbeat test cases
│   │   │   ├── test_models.py              <- API model test
│   │   │   ├── test_prediction.py          <- Test prediction data transfer
│   │   │   ├── test_sentencescoring.py     <- API sentence scoring test case
│   │   │   ├── test_sentmodel.py           <- API sentence scoring model test case
│   │   │   └── test_to_sentences.py        <- API sentence scoring payload test
│   │   ├── test_service
│   │   │   ├──test_models.py               <- ML model test cases
│   │   │   └──test_to_sent_model.py        <- Test sentence scoring returns
│   │   ├── conftest.py                     <- Test configuration / bootstrap test_client
│   └── └── README_TEST.md                  <- README specific for tests
│
├── env.example        <- Environment variables go here, can be read by `python-dotenv` package:
├── LICENSE            <- License file
├── README.md          <- The top-level README for developers using this project (this file)
├── requirements01.txt   <- General requirements loaded in requirements file
├── requirements02.txt   <- Torch specific requirements loaded in requirements file
├── requirements.txt   <- The requirements file for reproducing the analysis environment - consider using yml for conda environments
├── setup.py           <- Make this project pip installable with `pip install -e`
└── test_environment.py  <- verify environment is correct (dependencies), template only tests for python2 vs python3, feel free to add stuff
```
Updated: 30.06.2022