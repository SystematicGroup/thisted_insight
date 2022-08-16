from typing import List
from pydantic import BaseModel
class SentenceScoringPayloadSentences(BaseModel):
    sentence: str
    observations: List[str]

class SentenceScoringPayload(BaseModel):
    username: str
    text:str
    observations: List [str]
    sentences: List [SentenceScoringPayloadSentences]