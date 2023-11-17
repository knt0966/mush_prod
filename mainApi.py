# Install FastAPI and uvicorn if not installed
# pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
from fastapi.staticfiles import StaticFiles


# Load the model
model_dictionary = joblib.load('gpr_model_dictionary.pkl')
scaler_X = model_dictionary['scaler_X']
scaler_y = model_dictionary['scaler_y']
gp_model = model_dictionary['gp_model']

app = FastAPI()

static_dir = Path(__file__).resolve().parent / 'static'
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class InputData(BaseModel):
    bags: float
    temp: float
    hum: float

@app.post("/predict")
def predict_weight(input_data: InputData):
    # Scale the input data for prediction
    input_data_scaled = scaler_X.transform([[input_data.bags, input_data.temp, input_data.hum]])

    # Make the prediction
    predicted_weight_scaled = gp_model.predict(input_data_scaled)

    # Inverse transform the predicted value to the original scale
    predicted_weight = scaler_y.inverse_transform(predicted_weight_scaled.reshape(-1, 1)).flatten()[0]

    return {"predicted_weight": predicted_weight, "unit": "g"}
