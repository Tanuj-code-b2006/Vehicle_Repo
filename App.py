import os
import urllib.request
from pathlib import Path
import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import Input 
import numpy as np
import cv2
import pandas as pd







# Page config
st.set_page_config(page_title="🚗 Vehicle Damage Classifier", page_icon="🚘", layout="wide")

# ============ MODEL DOWNLOAD ============

MODEL_PATH = "car_damage_model_new.keras"
MODEL_URL = "https://github.com/Tanuj-code-b2006/car-damage-model/raw/main/car_damage_model_new.keras"
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Model download ho raha hai... 170MB hai, 1-2 min lagega ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model download ho gaya! ✅")

    # Yahi asli fix hai
    # 1. Pehle model ka structure bina weights ke banao
    # 2. Phir sirf weights load karo
    
    # Input layer khud se bana do
    inputs = Input(shape=(224, 224, 3)) # <-- apna input size yahi daalna
    
    # Model ko safe_mode=False ke saath load karo
    model = load_model(MODEL_PATH, compile=False, safe_mode=False)
    
    return model

model = load_my_model()
# ========================

# Labels mapping
class_labels = {
    0: "Dent",
    1: "Scratch",
    2: "Broken Glass",
    3: "Total Loss",
    4: "No Damage"
}

# Insurance claim mapping
insurance_claim = {
    "Dent": "Eligible - 40% claim",
    "Scratch": "Eligible - 20% claim",
    "Broken Glass": "Eligible - 60% claim",
    "Total Loss": "Eligible - 100% claim",
    "No Damage": "Not Eligible - 0% claim"
}

# Sidebar for user info
st.sidebar.header("🧑 Driver & 🚘 Vehicle Information")
driver_age = st.sidebar.number_input("Driver Age", min_value=18, max_value=100, value=35)
driver_gender = st.sidebar.selectbox("Driver Gender", ["Male", "Female", "Other"])
vehicle_make = st.sidebar.text_input("Vehicle Make", "Toyota")
vehicle_model = st.sidebar.text_input("Vehicle Model", "Camry")
vehicle_year = st.sidebar.number_input("Vehicle Year", min_value=1990, max_value=2026, value=2018)
vehicle_value = st.sidebar.number_input("Vehicle Value ($)", min_value=1000, value=20000)
accident_history = st.sidebar.number_input("Accident History", min_value=0, value=1)
policy_type = st.sidebar.selectbox("Policy Type", ["Comprehensive", "Third-Party"])
policy_coverage = st.sidebar.number_input("Policy Coverage ($)", min_value=1000, value=25000)
region = st.sidebar.text_input("Region", "California")
urban_area = st.sidebar.checkbox("Urban Area", value=True)
average_speed = st.sidebar.number_input("Average Speed (mph)", min_value=0, value=45)
braking_incidents = st.sidebar.number_input("Braking Incidents", min_value=0, value=3)

# Main UI
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚗 Vehicle Damage Classification</h1>", unsafe_allow_html=True)
st.write("Upload an image of your vehicle to check damage prediction")

uploaded_file = st.file_uploader("📤 Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Uploaded Vehicle Image", use_container_width=True)

    # Resize to model input size
    input_shape = model.input_shape[1:3]
    img_resized = cv2.resize(img, input_shape) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Prediction
    prediction = model.predict(img_resized)
    pred_class = np.argmax(prediction, axis=1)[0]
    damage_status = class_labels[pred_class]
    claim_status = insurance_claim[damage_status]

    # Stylish prediction output
    st.markdown(f"<h2 style='color: #27AE60;'>✅ Status: {damage_status}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #E74C3C;'>💰 Insurance Claim: {claim_status}</h3>", unsafe_allow_html=True)

    # Show user metadata in table
    st.markdown("---")
    st.subheader("📋 Driver & Vehicle Details")
    info_dict = {
        "Driver Age": driver_age,
        "Driver Gender": driver_gender,
        "Vehicle Make": vehicle_make,
        "Vehicle Model": vehicle_model,
        "Vehicle Year": vehicle_year,
        "Vehicle Value ($)": vehicle_value,
        "Accident History": accident_history,
        "Policy Type": policy_type,
        "Policy Coverage ($)": policy_coverage,
        "Region": region,
        "Urban Area": urban_area,
        "Average Speed (mph)": average_speed,
        "Braking Incidents": braking_incidents,
        "Damage Status": damage_status,
        "Insurance Claim": claim_status
    }
    st.table(pd.DataFrame(info_dict.items(), columns=["Attribute", "Value"]))