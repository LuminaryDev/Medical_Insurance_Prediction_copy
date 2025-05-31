import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('insurance_charges_model.pkl')

# Load the scaler (make sure you save it during training as well if you haven't)
scaler = joblib.load('insurance_scaler.pkl') # Assuming you saved your StandardScaler

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
    numerical_cols = ['age', 'bmi', 'children']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df[features]

def main():
    st.title('Medical Insurance Charges Prediction')
    st.write('Enter the patient details to predict insurance charges.')

    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    sex = st.selectbox('Sex', options=['female', 'male'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=5, value=0)
    smoker = st.selectbox('Smoker', options=['no', 'yes'])
    region = st.selectbox('Region', options=['southwest', 'southeast', 'northwest', 'northeast'])

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
        prediction = model.predict(processed_data)
        predicted_charge = np.expm1(prediction)[0] if you log-transformed charges else prediction[0] # Revert log transformation if applied

        st.subheader('Predicted Insurance Charges:')
        st.write(f'${predicted_charge:,.2f} USD')

if __name__ == '__main__':
    main()