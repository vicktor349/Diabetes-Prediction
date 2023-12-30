import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating function for prediction

def diabetes_prediction(input_data):

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "This patient is not diabetic"
    else:
        return "This patient is diabetic"
    

def main():
    """Main script for running the web app"""
    # Giving a title
    st.title("Diabetes Prediction Web App")

    # Getting the input data from the user
    Pregnancies = st.text_input("Enter the number of pregnancies")
    Glucose = st.text_input("Enter your glucose level")
    BloodPressure = st.text_input("Enter your blood pressure level")
    SkinThickness = st.text_input("Enter skin thickness value")
    Insulin = st.text_input("Enter your insulin level")
    BMI = st.text_input("Enter BMI Value")
    DiabetesPedigreeFunction = st.text_input("Enter Diabetes Pedigree Function Value")
    Age = st.text_input("Enter your age")


    # Code for prediction
    diagnosis = ''

    # Button a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)



if __name__ == '__main__':
    main()