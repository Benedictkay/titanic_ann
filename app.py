import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the trained mdel and any necessary preprocessing steps
# model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
feat_names = joblib.load('columns.pkl')
model_ann = load_model('titanic.keras')

# Set up the Streamlit app interface

st.title("Titanic Survival Prediction App")
st.header("Welcome to the Titanic Survival Prediction App!")
st.text("This app allows you to predict whether a passenger survived the Titanic disaster based on their features. Please enter the passenger's details below and click the 'Predict' button to see the result.")

# Create input fields for user to enter passenger details
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.slider("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Passenger Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)", ["C", "Q", "S"])
alone = st.selectbox("Alone", ["Yes", "No"])

# Create a button to trigger the prediction
if st.button("Predict Survival"):
    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        "sex": le.transform([sex])[0],
        "alone": le.transform([alone])[0],
        # One-hot encode the 'embarked' feature
        "embarked_C": 1 if embarked == 'C' else 0,
        "embarked_Q": 1 if embarked == 'Q' else 0,
        "embarked_S": 1 if embarked == 'S' else 0,
        "pclass": pclass,
        "age": age,  
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare
    }, index=[0])

    
    input_data = input_data[scaler.feature_names_in_] # Ensure the input data has the same feature order as the training data

    # Scale the input data using the same scaler used during training
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction using the trained model
    prediction_ann = model_ann.predict(input_data_scaled)

    # Display the prediction result
    st.write(f"Prediction probability: {prediction_ann[0][0]}")
    if prediction_ann[0][0] > 0.5:
        st.success('The passenger is predicted to survive.')
    else:
        st.error('The passenger is predicted not to survive.')



