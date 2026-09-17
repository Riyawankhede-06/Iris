import streamlit as st
import numpy as np
import joblib

#load model
model = joblib.load("iris_model.pkl")

#page title
st.title("machine learning on iris dataset")

#inputs labels
sepal_length = st.number_input("sepal_length")
sepal_width = st.number_input("sepal_width")
petal_length = st.number_input("petal_length")
petal_width = st.number_input("petal_width")

#prediction
if st.button("predict"):
  input = np.array([[ sepal_length,
  sepal_width,petal_length,petal_width ]]).astype(np.float64)
  prediction = model.predict(input)
  st.write(prediction)
