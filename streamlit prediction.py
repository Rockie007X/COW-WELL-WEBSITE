import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

# Load dataset
data = pd.read_csv(
    r"E:\live in lab contents\CODE\Cattle-disease-prediction-using-Machine-Learning-main\Training.csv",
    encoding="utf-8"
)

X = data.drop("prognosis", axis=1)
y = data["prognosis"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit UI
st.set_page_config(page_title="Cattle Disease Prediction", layout="centered")
st.title("🐄 நோய் முன்னறிவிப்பு முறைமை (Cattle Disease Predictor)")

st.markdown(f"**📊 மாதிரி துல்லியம் (Model Accuracy): {accuracy * 100:.2f}%**")

# Input Section
st.subheader("🩺 அறிகுறிகளை உள்ளிடவும்")

symptoms = st.multiselect("அறிகுறிகள் தேர்ந்தெடுக்கவும் (Select Symptoms)", X.columns.tolist())
temperature = st.number_input("வெப்பநிலை (°C):", 30.0, 45.0, step=0.1)
weight = st.number_input("எடை (Kg):", 50.0, 800.0, step=1.0)
pulse_rate = st.number_input("துடிப்பு விகிதம் (bpm):", 40.0, 180.0, step=1.0)

# Predict Button
if st.button("🔍 நோய் கணிக்க"):
    input_data = [1 if col in symptoms else 0 for col in X.columns]
    input_df = pd.DataFrame([input_data], columns=X.columns)

    predicted_index = model.predict(input_df)[0]
    predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

    st.success(f"✅ கணிக்கப்பட்ட நோய்: **{predicted_disease}**")
    st.info(f"🌡 வெப்பநிலை: {temperature}°C\n⚖ எடை: {weight} Kg\n❤️ துடிப்பு விகிதம்: {pulse_rate} bpm")

    # Show disease image if available
    image_path = os.path.join(
        r"E:\live in lab contents\CODE\Cattle-disease-prediction-using-Machine-Learning-main\disease_images",
        f"{predicted_disease}.jpg"
    )
    if os.path.exists(image_path):
        st.image(Image.open(image_path), caption=predicted_disease, use_container_width=True)
    else:
        st.warning("படம் இல்லை: இந்த நோய்க்கான படம் கிடைக்கவில்லை.")
