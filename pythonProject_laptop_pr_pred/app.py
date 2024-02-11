import streamlit as st
import pickle
import pandas as pd
import numpy as np

# importing the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pd.read_pickle('data.pkl')
st.title('Laptop Price Predictor')

# brand
company = st.selectbox('Brand', data['Company'].unique())
# type of the laptop
Type = st.selectbox('Type', data['TypeName'].unique())
# Ram
Ram = st.selectbox('RAM(In GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
Weight = st.number_input('Laptop Weight')
# Touchscreen
Touchscreen = st.selectbox('Touchscreen', ['NO', 'Yes'])
# IPS
IPS = st.selectbox('IPS', ['NO', 'Yes'])
# screen size
Screen_size = st.number_input('Screen Size')
# Resolution
Resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '1600x900',
                                                '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440',
                                                '2304x1440'])
# cpu
Cpu = st.selectbox('CPU', data['Cpu brand'].unique())
# HDD
HDD = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
# ssd
ssd = st.selectbox('ssd(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
Gpu = st.selectbox('GPU', data['Gpu brand'].unique())
# os
OS = st.selectbox('OS', data['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    if IPS == 'Yes':
        IPS = 1
    else:
        IPS = 0

    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / Screen_size
    query = np.array([company, Type, Ram, Weight, Touchscreen, IPS, ppi, Cpu, HDD, ssd, Gpu, OS])

    query = query.reshape(1, 12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
