# Observation Insight API

## API description

Start the application as detailed in [README.md](../../README.md) then you may call the following APIs

### Heartbeat
This call takes no arguments and only serve to test if application is still running

### Prediction
See [prediction.py](../../observation_insight/app/api/routes/prediction.py)

Takes [PredictionPayload](../../observation_insight/app/models/payload.py) class as input. This is a [pydantic](https://pydantic-docs.helpmanual.io/) wrapper for the following JSON input
```
{"text": "this is a test text - you should fill in something more meaningful"}
```

The result is also returned as JSON in the form
```
{"result": [
    {"name": "obs1", "probability": 0.9},
    {"name": "obs2", "probability": 0.8}
    ]}
```
Where `obs1`, `obs2` will be valid human readable names.

This can be tested from command line using [curl](https://curl.se/) as for instance (on linux)
```
curl -X "POST"   http://127.0.0.1:8000/api/predict   -H "accept: application/json"   -H "Content-Type: application/json"   -d "{\"text\": \"this is a test text - you should fill in something more meaningful\" }"
```
Where `YOURURL` often will be <http://127.0.0.1:8000> when running locally
