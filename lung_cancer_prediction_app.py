import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load your trained KNN model and scaler
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Lung Cancer Risk Prediction")

# Input form
patient_id = st.number_input("Patient Id", min_value=0, max_value=1000000, value=0)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
air_pollution = st.slider("Air Pollution Exposure", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
alcohol_use = st.slider("Alcohol Use", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
dust_allergy = st.slider("Dust Allergy", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
occupational_hazards = st.slider("Occupational Hazards", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
genetic_risk = st.slider("Genetic Risk", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
chronic_lung_disease = st.slider("Chronic Lung Disease", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
balanced_diet = st.slider("Balanced Diet", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
obesity = st.slider("Obesity", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
smoking = st.slider("Smoking", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
passive_smoker = st.slider("Passive Smoker", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
chest_pain = st.slider("Chest Pain", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
coughing_of_blood = st.slider("Coughing of Blood", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
fatigue = st.slider("Fatigue", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
weight_loss = st.slider("Weight Loss", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
shortness_of_breath = st.slider("Shortness of Breath", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
wheezing = st.slider("Wheezing", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
swallowing_difficulty = st.slider("Swallowing Difficulty", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
clubbing_of_finger_nails = st.slider("Clubbing of Finger Nails", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
frequent_cold = st.slider("Frequent Cold", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
dry_cough = st.slider("Dry Cough", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
snoring = st.slider("Snoring", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# Create a dataframe for the input
input_data = {
    'Patient Id': patient_id,
    'Age': age,
    'Gender': 1 if gender == "Male" else 2 if gender == "Female" else 3,
    'Air Pollution': air_pollution,
    'Alcohol use': alcohol_use,
    'Dust Allergy': dust_allergy,
    'OccuPational Hazards': occupational_hazards,
    'Genetic Risk': genetic_risk,
    'chronic Lung Disease': chronic_lung_disease,
    'Balanced Diet': balanced_diet,
    'Obesity': obesity,
    'Smoking': smoking,
    'Passive Smoker': passive_smoker,
    'Chest Pain': chest_pain,
    'Coughing of Blood': coughing_of_blood,
    'Fatigue': fatigue,
    'Weight Loss': weight_loss,
    'Shortness of Breath': shortness_of_breath,
    'Wheezing': wheezing,
    'Swallowing Difficulty': swallowing_difficulty,
    'Clubbing of Finger Nails': clubbing_of_finger_nails,
    'Frequent Cold': frequent_cold,
    'Dry Cough': dry_cough,
    'Snoring': snoring
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the inputs
input_df_scaled = scaler.transform(input_df.drop(columns=['Patient Id']))

# Predict the risk level
risk_prediction = knn_model.predict(input_df_scaled)

st.write(f"The predicted lung cancer risk level for Patient Id {patient_id} is: {risk_prediction[0]}")
