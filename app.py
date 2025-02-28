from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from test_cost import combine_prediction_and_estimation

# Set up FastAPI
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the TensorFlow Lite model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image preprocessing settings
img_size = (256, 256)  # Must match training size
categories = ["group1", "group2", "group3", "group4"]  # Labels

# Hair Transplant Cost Table
COUNTRIES = {
    'United States': {'low': 2, 'high': 10},
    'Canada': {'low': 1.85, 'high': 5.18},
    'United Kingdom': {'low': 4.88, 'high': 4.88},
    'Australia': {'low': 1.28, 'high': 2.94},
    'Singapore': {'low': 2.88, 'high': 4.32},
    'Turkey': {'low': 0.58, 'high': 2.63},
    'India': {'low': 0.78, 'high': 0.78}
}

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    file: UploadFile = File(...),
    age: int = Form(...),
    smokes: bool = Form(...),
    race: str = Form(...)
):
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(await file.read())).resize(img_size)
        image = image.convert("L")
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Single-channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        
        expected_shape = input_details[0]['shape']
        if img_array.shape != tuple(expected_shape):
            raise ValueError(f"Input shape mismatch: Expected {expected_shape}, got {img_array.shape}")

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = categories[np.argmax(predictions)]

        # Convert category to group number
        group_num = int(predicted_class[-1])  # Extract last digit from category name
        
        # Calculate Norwood Stage & Grafts
        norwood_stage, grafts_needed = combine_prediction_and_estimation(group_num, age, smokes, race)

        # Generate cost estimates
        cost_table = ""
        if isinstance(grafts_needed, int):
            cost_table += "<h3>Estimated Hair Transplant Costs</h3>"
            cost_table += "<table border='1'><tr><th>Country</th><th>Low Cost (USD)</th><th>High Cost (USD)</th><th>Avg Cost (USD)</th></tr>"

            for country, prices in COUNTRIES.items():
                low_cost = prices['low'] * grafts_needed
                high_cost = prices['high'] * grafts_needed
                avg_cost = ((prices['low'] + prices['high']) / 2) * grafts_needed

                cost_table += f"<tr><td>{country}</td><td>${low_cost:,.0f}</td><td>${high_cost:,.0f}</td><td>${avg_cost:,.0f}</td></tr>"
            
            cost_table += "</table>"
        else:
            cost_table = f"<h3>{grafts_needed}</h3>"

        # Generate HTML response
        html_response = f"""
        <html>
        <head><title>Hair Loss Prediction</title></head>
        <body>
            <h4>Prediction Result</h4>
            <p><strong>Predicted Class:</strong> {predicted_class}</p>
            <p><strong>Norwood Stage:</strong> {norwood_stage}</p>
            {cost_table}
        </body>
        </html>
        """
        return HTMLResponse(content=html_response)
    
    except Exception as e:
        return HTMLResponse(content=f"<h3>Error: {str(e)}</h3>")

@app.get("/", response_class=HTMLResponse)
async def home():
    home_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "home.html")
    with open(home_path, "r", encoding="utf-8") as file:
        return file.read()
