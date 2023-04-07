import streamlit as st
from keras.models import load_model
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.set_page_config('Diabetes Diagnosis', ':ambulance:')
st.title('Diabetes Diagnosis :ambulance:')

folder_path = os.path.dirname(__file__)
model = load_model(os.path.join(folder_path, 'model.h5'))
scaler = MinMaxScaler()

def predict():
    features = np.array([[pregnancies, glucose, bp, thickness, insulin, bmi, dpf, age]])
    scaled_features = scaler.fit_transform(features.T)
    print(scaled_features)
    prob = model.predict(scaled_features.T, verbose=False)
    print(prob)
    res = np.round(prob)[0][0]

    if res == 0:
        st.success('You are free from Diabetes!')
    else:
        st.error('You are diagnosed to have diabetes! Visit a Doctor!')

pregnancies = st.number_input('Number of times pregnant', step=1)
glucose = st.number_input('Plasma glucose concentration', step=1)
bp = st.number_input('Diastolic blood pressure (mm Hg)', step=1)
thickness = st.number_input('Triceps skin fold thickness (mm)', step=1)
insulin = st.number_input('2-Hour serum insulin (mu U/ml)', step=1)
bmi = st.number_input('Body mass index', step=0.1)
dpf = st.number_input('Diabetes pedigree function', step=0.01)
age = st.number_input('Age (years)', step=1)

st.button('Diagnose', on_click=predict)
