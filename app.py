import gradio as gr
import pandas as pd
import joblib

# Load the trained XGBoost model
model = joblib.load('xgboost-model.pkl')  # make sure the model file is present

# Define the prediction function
def predict_survival(age, anaemia, creatinine_phosphokinase, diabetes,
                     ejection_fraction, high_blood_pressure, platelets,
                     serum_creatinine, serum_sodium, sex, smoking, time):

    input_data = pd.DataFrame({
        'age': [age],
        'anaemia': [anaemia],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [high_blood_pressure],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [sex],
        'smoking': [smoking],
        'time': [time]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "⚠️ Patient did not survive during follow-up."
    else:
        return "✅ Patient survived during follow-up."

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Slider(30, 100, step=1, label="Age (years)"),
        gr.Radio([0, 1], label="Anaemia (0: No, 1: Yes)"),
        gr.Slider(20, 8000, step=10, label="Creatinine Phosphokinase (mcg/L)"),
        gr.Radio([0, 1], label="Diabetes (0: No, 1: Yes)"),
        gr.Slider(10, 80, step=1, label="Ejection Fraction (%)"),
        gr.Radio([0, 1], label="High Blood Pressure (0: No, 1: Yes)"),
        gr.Slider(50000, 850000, step=1000, label="Platelets (kiloplatelets/mL)"),
        gr.Slider(0.5, 10.0, step=0.1, label="Serum Creatinine (mg/dL)"),
        gr.Slider(110, 150, step=1, label="Serum Sodium (mEq/L)"),
        gr.Radio([0, 1], label="Sex (0: Female, 1: Male)"),
        gr.Radio([0, 1], label="Smoking (0: No, 1: Yes)"),
        gr.Slider(1, 300, step=1, label="Follow-up Time (days)")
    ],
    outputs="text",
    title="💓 Heart Failure Survival Prediction App",
    description="Provide patient's clinical data to predict survival after heart failure."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
