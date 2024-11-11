import streamlit as st
import numpy as np
import joblib

# Load the trained model 
rf_model = joblib.load('heart_disease_rf_model.pkl')

# Function to predict heart disease
def predict_heart_disease(age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar,
                           restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, 
                           noofmajorvessels):
    # Map chestpain categories to numeric values
    chestpain_mapping = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    chestpain = chestpain_mapping.get(chestpain, 0)  # Default to 0 if no valid input
    
    # Map 'Yes'/'No' to numeric values for fastingbloodsugar and exerciseangia
    fastingbloodsugar = 1 if fastingbloodsugar == "Yes" else 0
    exerciseangia = 1 if exerciseangia == "Yes" else 0
    
    # Map slope categories to numeric values
    slope_mapping = {
        "Upsloping": 0,
        "Flat (Horizontal)": 1,
        "Downsloping": 2
    }
    slope = slope_mapping.get(slope, 0)  # Default to 0 if no valid input
    
    # Prepare input data as a numpy array (reshaped to match the expected input shape for the model)
    input_data = np.array([[age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar,
                            restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels]])
    
    # Predict using the trained RandomForest model
    prediction = rf_model.predict(input_data)
    return prediction[0]

# Streamlit app interface
def main():
    st.title("Heart Disease Prediction")
    st.write("Enter the details of the patient to predict the likelihood of heart disease.")

    # Input fields for all the required features
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    chestpain = st.selectbox("Chest Pain Type", 
                             options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    restingBP = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=0, value=120)
    serumcholestrol = st.number_input("Serum Cholestrol (in mg/dl)", min_value=0, value=200)
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])  # Updated
    restingrelectro = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])  # Categorical
    maxheartrate = st.number_input("Max Heart Rate Achieved", min_value=0, value=150)
    exerciseangia = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])  # Updated
    oldpeak = st.number_input("Old Peak (ST depression)", min_value=0.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                         options=["Upsloping", "Flat (Horizontal)", "Downsloping"])  # Updated
    noofmajorvessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])  # Categorical

    # When the user clicks the predict button
    if st.button("Predict Heart Disease"):
        # Convert gender to numerical format: Male = 1, Female = 0 (or vice versa)
        gender = 1 if gender == "Male" else 0
        
        # Get prediction
        result = predict_heart_disease(age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar,
                                       restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels)
        
        # Display prediction result
        if result == 1:
            st.write("The model predicts: **Heart Disease Detected**")
        else:
            st.write("The model predicts: **No Heart Disease Detected**")

if __name__ == "__main__":
    main()
