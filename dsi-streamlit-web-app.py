#IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import joblib


#LOAD OUR MODEL PIPELINE OBJECT
model = joblib.load("model.joblib")

#ADD TITLE AND INSTRUCTIONS
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

#AGE INPUT FORM
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)

#GENDER INPUT FORM
gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ["M","F"])

#CREDIT SCORE INPUT FORM
credit_score = st.number_input(
    label = "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500)

#SUBMIT INPUTS TO MODEL
if st.button("Submit For Prediction"):
    
    #store our data in a dataframe for prediction
    new_data = pd.DataFrame({"age":[age], "gender": [gender], "credit_score": [credit_score]})

    #Apply Model Pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    #Output prediction 
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")
    

