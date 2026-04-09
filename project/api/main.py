from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference.predict import predict

app = FastAPI()


class PredictRequest(BaseModel):
    age: int = Field(..., gt=0)
    sex: int
    cp: int
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., gt=0)
    fbs: int
    restecg: int
    thalach: float = Field(..., gt=0)
    exang: int
    oldpeak: float = Field(..., ge=0)
    slope: int
    ca: int
    thal: int

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145.0,
                "chol": 233.0,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150.0,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }


@app.post("/predict")
async def predict_endpoint(payload: PredictRequest):
    try:
        payload_dict = payload.dict()
        result = predict(payload_dict)
        return result if result.get("success") else result
    except Exception as exc:
        return {"error": str(exc), "success": False}


@app.get("/health")
def health():
    return {"status": "ok"}
