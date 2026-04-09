# API Contract: Heart Disease Prediction Service

## Endpoint
`POST /predict`

## Request Body
All fields are required JSON properties.

## Fields
- `age`: integer, > 0
- `sex`: integer, in {0, 1}
- `cp`: integer, in {0, 1, 2, 3}
- `trestbps`: integer, > 0
- `chol`: integer, > 0
- `fbs`: integer, in {0, 1}
- `restecg`: integer, in {0, 1, 2}
- `thalach`: integer, > 0
- `exang`: integer, in {0, 1}
- `oldpeak`: float, >= 0.0
- `slope`: integer, in {0, 1, 2}
- `ca`: integer, in {0, 1, 2, 3, 4}
- `thal`: integer, in {0, 1, 2, 3}

## Example Request
```json
{
  "age": 57,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 236,
  "fbs": 0,
  "restecg": 0,
  "thalach": 174,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 1,
  "ca": 1,
  "thal": 2
}
```

## Success Response
```json
{
  "prediction": 1,
  "probability": 0.78,
  "threshold_used": 0.30,
  "success": true
}
```
Note: The values above are an example. The threshold is dynamically loaded and may differ.

## Error Response
```json
{
  "error": "Invalid input: age must be > 0",
  "success": false
}
```

## Prediction Meaning
- `0` = no disease
- `1` = high risk

## Threshold Behavior
- Threshold is dynamically loaded from `artifacts`.
- Decision is based on `probability >= threshold`.
- The threshold value is dynamically loaded from `artifacts/threshold.json` and may change after retraining.

## Backend Responsibilities
- Validate JSON format before sending.
- Send request to AI service.
- Handle success and error responses.

## AI Responsibilities
- Validate input.
- Load model.
- Apply preprocessing.
- Generate prediction.
- Return structured response.
