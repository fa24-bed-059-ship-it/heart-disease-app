import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

categories = {
    "Sex": ["Male", "Female"],
    "ChestPain": ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
}
encoders = {}
for column, values in categories.items():
    enc = LabelEncoder()
    enc.fit(values)
    encoders[column] = enc

@st.cache_resource
def load_model():
    with open("heart_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("❤️ Heart Disease Prediction System")
age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
sex = st.selectbox("Sex", categories["Sex"])
chest_pain = st.selectbox("Chest pain type", categories["ChestPain"])
max_hr = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    user_input = pd.DataFrame({"Age": [age], "Sex": [sex],
                               "ChestPain": [chest_pain], "MaxHeartRate": [max_hr]})
    for column in categories:
        user_input[column] = encoders[column].transform(user_input[column])
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.error("Prediction: HEART DISEASE LIKELY")
    else:
        st.success("Prediction: NO HEART DISEASE")
