import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title('IRIS flower Prediction App')
st.write('This App predicts the IRIS flower type given the user inputs')

st.sidebar.header('User Input')

def user_input():
    sepal_length = st.sidebar.slider('Sepal Length',4.3,7.9,5.2)
    sepal_width = st.sidebar.slider('Sepal Width',2.0,4.4,3.2)
    petal_length = st.sidebar.slider('Petal Length',1.0,6.9,3.2)
    petal_width = st.sidebar.slider('Petal Width',0.1,2.5,1.2)
    data = {
    'Sepal Length' : sepal_length,
    'Sepal Width' : sepal_width,
    'PetalLength' : petal_length,
    'Petal Width' : petal_width,
    }
    features = pd.DataFrame(data,index = [0])
    return features

df = user_input()

# show dataTable
st.subheader('Data Table')
st.write(df)


#loading dataset and usinf RandomForestClassifier
iris= datasets.load_iris()
X = iris.data
Y = iris.target

#rfc
rfc = RandomForestClassifier()
rfc.fit(X,Y)

#Prediction
Prediction  = rfc.predict(df)
Prediction_Probability = rfc.predict_proba(df)


#class labels
st.subheader('Class Labels and their index numbers')
st.write(iris.target_names)

# Display Prediction Probability
st.subheader('Prediction Probability')
st.write(Prediction_Probability)

# display Prediction
st.subheader('Prediction')
st.write(iris.target_names[Prediction])
