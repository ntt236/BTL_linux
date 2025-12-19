from pydantic import BaseModel, Field

class PredictReq(BaseModel):
    math: float = Field(ge=0, le=10)
    physics: float = Field(ge=0, le=10)
    chemistry: float = Field(ge=0, le=10)
    english: float = Field(ge=0, le=10)
    priority: float = Field(ge=0, le=2)

class PredictRes(BaseModel):
    probability: float
    percent: float
    label: str
    model_version: str
