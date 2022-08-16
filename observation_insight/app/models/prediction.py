from pydantic import BaseModel

class PredictionResult(BaseModel):
    output_feature: int