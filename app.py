import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to make predictions using the loaded model
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age],
        'Is_Obese': [1 if BMI >= 30 else 0]
    })
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Return the prediction result
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Create a Gradio interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="DiabetesPedigreeFunction"),
        gr.Number(label="Age")
    ],
    outputs=gr.Textbox(label="Prediction")
)

# Launch the interface
interface.launch(share=True) 