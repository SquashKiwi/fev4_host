import time
import streamlit as st
import numpy as np
import tensorflow as tf
from test_cost import combine_prediction_and_estimation
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- GLOBAL CONFIG ---
IMG_SIZE = (256, 256)
CATEGORIES = ["group1", "group2", "group3", "group4"]

# --- FUNCTIONS ---
def load_model(model_filename: str):
    interpreter = tf.lite.Interpreter(model_path=model_filename)
    interpreter.allocate_tensors()

    # Get input and output details
    return interpreter

def predict_image(interpreter, image_path: str):
    """Predict the class of a given image using the loaded model."""
    try:
        img = load_img(image_path, target_size=IMG_SIZE, color_mode="grayscale")
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        return CATEGORIES[predicted_idx], confidence
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def refined_norwood_prediction(group: int, age: int, smokes: bool, race: str) -> str:
    """Refines Norwood stage prediction based on age, smoking, and race."""
    normalized_age = (age - 39.67) / 17.73
    race_scores = {'Asian': 0.183, 'Caucasian': -0.04, 'Black': -0.145}
    
    base_score = normalized_age * 0.88 + (0.64 if smokes else 0) + race_scores.get(race, 0)
    thresholds = {1: -0.92, 2: -0.5, 3: 1.0, 4: 1.75}

    if group == 1:
        return "Norwood Stage 1"
    elif group == 2:
        return "Norwood Stage 3" if base_score > thresholds[2] else "Norwood Stage 2"
    elif group == 3:
        return "Norwood Stage 5" if base_score > thresholds[3] else "Norwood Stage 4"
    elif group == 4:
        return "Norwood Stage 7" if base_score > thresholds[4] else "Norwood Stage 6"
    else:
        return "Unknown Group"

def estimate_grafts(norwood_stage: str, age: int) -> int:
    """Estimate the number of grafts needed based on Norwood stage and age."""
    graft_ranges = {
        "II": (800, 1200), "III": (1000, 1500), "IV": (1600, 2200),
        "V": (2000, 2700), "VI": (2500, 3500), "VII": (3500, 5000)
    }
    if norwood_stage not in graft_ranges:
        return None
    
    min_grafts, max_grafts = graft_ranges[norwood_stage]
    interpolation_factor = min(max((age - 25) / (70 - 25), 0), 1)
    estimated_grafts = min_grafts + interpolation_factor * (max_grafts - min_grafts)
    
    return int(round(estimated_grafts))

def combine_prediction_and_estimation(group: int, age: int, smokes: bool, race: str):
    """Combines the refined Norwood prediction with the graft estimation."""
    refined_stage_str = refined_norwood_prediction(group, age, smokes, race)
    match = re.search(r"(\d+)$", refined_stage_str)
    if not match:
        return refined_stage_str, None

    numeric_stage = int(match.group(1))
    norwood_map = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}
    roman_stage = norwood_map.get(numeric_stage)

    if not roman_stage:
        return refined_stage_str, None

    graft_est = estimate_grafts(roman_stage, age)
    return refined_stage_str, graft_est

def print_cost_table(grafts):
    """Generates a cost estimation for different countries."""
    countries = {
        'United States': {'low': 2, 'high': 10},
        'Canada': {'low': 1.85, 'high': 5.18},
        'United Kingdom': {'low': 4.88, 'high': 4.88},
        'Australia': {'low': 1.28, 'high': 2.94},
        'Singapore': {'low': 2.88, 'high': 4.32},
        'Turkey': {'low': 0.58, 'high': 2.63},
        'India': {'low': 0.78, 'high': 0.78}
    }
    
    cost_table = []
    for country, prices in countries.items():
        low_cost = prices['low'] * grafts
        high_cost = prices['high'] * grafts
        avg_cost = ((prices['low'] + prices['high']) / 2) * grafts
        cost_table.append((country, f"${prices['low']} - ${prices['high']}", f"${low_cost:,.0f}", f"${high_cost:,.0f}", f"${avg_cost:,.0f}"))
    
    return cost_table

# --- STREAMLIT UI ---
st.title("Male Pattern Baldness Prediction & Cost Estimation")

# Upload image
uploaded_file = st.file_uploader("Upload a scalp image (JPG/PNG)", type=["jpg", "png", "jpeg"])

# User inputs
age = st.number_input("Enter Age", min_value=18, max_value=80, value=30)
smokes = st.checkbox("Smoker?")
race = st.selectbox("Select Race", ["Asian", "Caucasian", "Black"])
go = st.button('Go', type= "primary" if uploaded_file and age and race else "secondary")

if uploaded_file and go:
    with st.spinner("Processing image..."):
        time.sleep(1)
        model = load_model("model.tflite")  # Ensure the correct path
        if model:
            # Save image temporarily
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Predict group
            predicted_class, confidence = predict_image(model, temp_image_path)

            if predicted_class:
                group = int(re.sub(r"\D", "", predicted_class))
                refined_stage, grafts = combine_prediction_and_estimation(group, age, smokes, race)

                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                st.write(f"**Predicted Class:** {predicted_class} (Confidence: {confidence:.2f}%)")
                st.write(f"**Refined Norwood Stage:** {refined_stage}")

                if isinstance(grafts, int):
                    st.write(f"**Estimated Grafts Needed:** {grafts}")
                    st.write("### Cost Estimation:")
                    cost_data = print_cost_table(grafts)
                    st.table(cost_data)
                else:
                    st.write(f"**Message:** {grafts}")
            else:
                st.error("Prediction failed. Please try again with a different image.")
