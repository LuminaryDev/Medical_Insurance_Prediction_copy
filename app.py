import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# List files in the current directory
print("Current directory contents:")
print(os.listdir('.'))
print("--------------------")

# Load the trained model
model = joblib.load('titanic_lr_model.pkl')

# Feature names (make sure these match the order used during training)
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin',
                 'Embarked_Q', 'Embarked_S', 'FamilySize', 'Title_Miss',
                 'Title_Mr', 'Title_Mrs', 'Title_Rare']

def predict_survival(features):
    features_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)[:, 1]
    return prediction[0], probability[0]

def main():
    st.title('Titanic Survival Prediction')
    st.write('Enter passenger details to predict their survival.')

    # Create input fields for each feature
    pclass = st.sidebar.selectbox('Passenger Class', options=[1, 2, 3])
    sex = st.sidebar.selectbox('Sex', options=['male', 'female'])
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)
    sibsp = st.sidebar.number_input('SibSp (Number of siblings/spouses aboard)', min_value=0, max_value=10, value=0)
    parch = st.sidebar.number_input('Parch (Number of parents/children aboard)', min_value=0, max_value=10, value=0)
    fare = st.sidebar.number_input('Fare', min_value=0, max_value=600, value=30)
    has_cabin = st.sidebar.selectbox('Had Cabin?', options=['Yes', 'No'])
    embarked = st.sidebar.selectbox('Embarked Port', options=['Southampton', 'Queenstown', 'Cherbourg'])
    family_size = sibsp + parch + 1

    # Handle categorical features based on user input
    sex_encoded = 1 if sex == 'female' else 0
    has_cabin_encoded = 1 if has_cabin == 'Yes' else 0
    embarked_q = 1 if embarked == 'Queenstown' else 0
    embarked_s = 1 if embarked == 'Southampton' else 0

    # Assume a simple title based on Sex for demonstration. You might need more complex logic.
    title_mr = 1 if sex == 'male' else 0
    title_miss = 1 if sex == 'female' and age <= 30 else 0 # Simplified
    title_mrs = 1 if sex == 'female' and age > 30 else 0  # Simplified
    title_rare = 0

    features = [pclass, sex_encoded, age, sibsp, parch, fare, has_cabin_encoded,
                embarked_q, embarked_s, family_size, title_miss, title_mr, title_mrs, title_rare]

    if st.button('Predict Survival'):
        prediction, probability = predict_survival(features)

        st.subheader('Prediction:')
        if prediction == 1:
            st.write('The passenger is predicted to have survived.')
        else:
            st.write('The passenger is predicted not to have survived.')

        st.subheader('Probability of Survival:')
        st.write(f'{probability:.2f}')

if __name__ == '__main__':
    main()
