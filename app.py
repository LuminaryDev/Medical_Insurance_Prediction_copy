import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set to True if you log-transformed 'charges' during training, False otherwise
you_logged_charges = False  # Assuming you didn't based on the training notebook

# Load the trained model
try:
    model = joblib.load('insurance_charges_model.pkl')
except FileNotFoundError:
    st.error("Error: 'insurance_charges_model.pkl' not found. Make sure it's in the same directory or provide the correct path.")
    st.stop()

# Load the scaler (make sure you save it during training as well if you haven't)
try:
    scaler = joblib.load('insurance_scaler.pkl')
except FileNotFoundError:
    st.warning("Warning: 'insurance_scaler.pkl' not found. Ensure you saved it during training. The app might not function as expected without scaling.")
    scaler = None  # Handle the case where the scaler is not found

# Define features (order matters!)
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']

def preprocess(data):
    df = pd.DataFrame([data], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # Manual Binary Encoding for 'sex' and 'smoker'
    df['sex'] = df['sex'].map({'female': 1, 'male': 0}).fillna(0).astype(int)
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # One-Hot Encoding for 'region'
    region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True, dtype=int)
    df = pd.concat([df, region_dummies], axis=1)
    df.drop('region', axis=1, inplace=True)

    # Ensure all features are present and in the correct order
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0 # Or handle missing features appropriately

    # Select and scale numerical features
    # numerical_cols = ['age', 'bmi', 'children']
    # if scaler:
    #     df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df[features]

def main():
    st.title('Medical Insurance Charges Prediction')
    st.write('Enter the patient details to predict insurance charges.')

    age = st.number_input('Age', min_value=18, max_value=100, value=25)
    sex = st.selectbox('Sex', options=['female', 'male'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=22.5)
    children = st.number_input('Number of Children', min_value=0, max_value=5, value=0)
    smoker = st.selectbox('Smoker', options=['no', 'yes'])
    region = st.selectbox('Region', options=['northwest', 'southeast', 'southwest', 'northeast'])

    user_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }

    if st.button('Predict Charges'):
        processed_data = preprocess(user_data)
        st.write("Processed Data:", processed_data)  # For debugging
        prediction = model.predict(processed_data)
        predicted_charge = np.expm1(prediction)[0] if you_logged_charges else prediction[0] # Revert log transformation if applied

        st.subheader('Predicted Insurance Charges:')
        st.write(f'${predicted_charge:,.2f} USD')

if __name__ == '__main__':
    main()
