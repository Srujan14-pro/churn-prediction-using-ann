import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time

# -------------------------
# Load Model & Encoders
# -------------------------

model = tf.keras.models.load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)



# -------------------------
# Streamlit UI
# -------------------------

# Welcome Animation
st.balloons()
st.title(" Customer Churn Prediction App ")
st.markdown("""
<h4 style='color: #4CAF50;'>Welcome! Predict if a customer will churn and explore insights about your data.</h4>
""", unsafe_allow_html=True)

# Sidebar with info and fun fact
with st.sidebar:
    st.header("About App")
    st.info("This app predicts the probability of a customer churning using a trained neural network model.")
    st.markdown("---")
    st.subheader("💡 Fun Fact")
    st.write("Did you know? Retaining customers is 5x cheaper than acquiring new ones!")

st.write("Enter customer details below:")

# Categorical Inputs
geography = st.selectbox(
    "Geography",
    one_hot_encoder.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder.classes_
)

has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

# Numerical Inputs
credit_score = st.number_input("Credit Score", 300, 850, 600)
age = st.slider("Age", 18, 92, 30)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_of_products = st.number_input("Number of Products", 1, 4, 1)
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# -------------------------
# Prediction Button
# -------------------------


predict_btn = st.button("Predict")

if predict_btn:

    # Create DataFrame
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Geography': [geography],
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == "Yes" else 0],
        'IsActiveMember': [1 if is_active_member == "Yes" else 0],
        'EstimatedSalary': [estimated_salary]
    })

    # -------------------------
    # Encoding
    # -------------------------

    # Encode Gender
    input_data['Gender'] = label_encoder.transform(input_data['Gender'])

    # Encode Geography
    geo_encoded = one_hot_encoder.transform(
        input_data[['Geography']]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder.get_feature_names_out(['Geography'])
    )

    # Drop original Geography column
    input_data.drop(columns=['Geography'], inplace=True)

    # Combine encoded geography
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # -------------------------
    # Scaling
    # -------------------------

    # Ensure columns match scaler's expected order
    expected_columns = scaler.feature_names_in_
    input_data_ordered = input_data[expected_columns]
    input_data_scaled = scaler.transform(input_data_ordered)

    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    # -------------------------
    # Output
    # -------------------------


    st.write(f"Churn Probability: {prediction_prob:.2f}")

    if prediction_prob > 0.5:
        st.error("The customer is likely to churn.")
        st.markdown("<h4 style='color: #e74c3c;'>Take action to retain this customer!</h4>", unsafe_allow_html=True)
    else:
        st.success("The customer is unlikely to churn.")
        st.markdown("<h4 style='color: #27ae60;'>Great! Keep up the good work.</h4>", unsafe_allow_html=True)

    # Exit/Thank you message with confetti
    st.markdown("---")
    st.markdown("<h3 style='color: #2980b9;'>Thank you for using the Churn Prediction App!</h3>", unsafe_allow_html=True)
    st.snow()
    time.sleep(1)
    st.balloons()
