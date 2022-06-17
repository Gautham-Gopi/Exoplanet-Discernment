import streamlit as st
import pickle
import numpy as np
import pandas as pd
# from random import randint
# from streamlit import session_state


def load_model():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data

def mean_normalise(series):
  series = pd.Series(series)
  return (series - series.mean())/(series.max() - series.min())

def fast_fourier_transformation(series):
  series = np.fft.fft(series, len(series))
  return np.abs(series)

data=load_model()

classifier=data["model"]

def show_predict_page():
    st.title("Exo-Planet Prediction")

    st.write("""### We need flux data """)
    # state = get_session_state()
    # if not state.widget_key:
    #     state.widget_key = str(randint(1000, 100000000))
    uploaded_file = st.file_uploader(
    "Upload your csv file", type=["csv"], accept_multiple_files=False
    )
    col1, col2, col3 , col4 = st.columns(4)

    with col1:
        display=col1.button("Display Data")
    with col2:
        pass
    with col4:
        predict=col4.button("Predict")
    with col3 :
        pass
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        # df.drop(["LABEL"],axis=1,inplace=True)
        df=df.apply(mean_normalise,axis=1)
        df=df.T.apply(fast_fourier_transformation,axis=0).T

        if display:
            clear=st.button("Clear")  
            st.write(df)
            if clear:
                st.write("")
        if predict:        
            prediction=classifier.predict(df)
            if prediction[0]==0:
                st.subheader("Not a Exo-Planet")
            else:
                st.subheader("Exo-Planet")   
    # te
    # res=pd.read_json("uploaded_file")
