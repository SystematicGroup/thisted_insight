from typing import List
from pydantic import BaseModel


class PredictionPayload(BaseModel):
    text: str

def payload_to_list(hpp: PredictionPayload) -> List:
    return [
        hpp.text
        ]

# Sentence scoring
class SentPredictionPayload(BaseModel):
    username: str
    text: str
    obstype_chosen: List[str]

def sentpayload_to_dict(spp: SentPredictionPayload):
    return {
        "username": spp.username,
        "text": spp.text,
        "obstype_chosen": spp.obstype_chosen
    }