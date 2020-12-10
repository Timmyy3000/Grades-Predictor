import streamlit as st
import pandas as pd
from sklearn import metrics
import pickle


data = pd.read_csv('student-mat.csv', sep=";")

st.write("""
# Grade Predictor
### Aim
- Using a **Linear regression** model to predict a student's grade based on provided parameters.

Data provided from [*Student Performance Data Set*](https://archive.ics.uci.edu/ml/machine-learning-databases/00320/) from the [*UCI Machine Learning Repository*](https://archive.ics.uci.edu/).
""")


data = data[["G1", "G2" , "studytime", "failures", "absences", "G3"]]
st.subheader(' Data')
st.write("""
Randomly selected rows from dataset

*Hint : Input parameters from the table and compare results*
""")
st.write(data.sample(5))
st.write("""

Our Target Variable will be "G3"

""")


# SIDE BAR
st.sidebar.header('User Input Parameters')

def user_input_param():
    g1 = st.sidebar.slider('1st Grade', 0, 20, 5)
    g2 = st.sidebar.slider('2nd Grade', 0, 20, 6)
    study_time = st.sidebar.slider('Study Hours', 1, 5, 2)
    failures = st.sidebar.slider('Failures', 0, 5, 0)
    absences = st.sidebar.slider('Absences', 0, 50, 6)
    input = { 'G1' : g1,
              'G2' : g2,
              'studytime' : study_time,
              'failuers' : failures,
              'absences' : absences
              }
    param = pd.DataFrame(input, index=[0])
    return param

input_df = user_input_param()

# METRICS SUB HEADER

#splitting data
x_test = data.drop(['G3'], 1)
y_test = data['G3']

# load model
pickle_in = open("model.pickle", "rb")
model = pickle.load(pickle_in)

y_pred = model.predict(x_test)

#Getting Metrics
metrics = pd.Series({
'Accuracy' :   model.score(x_test, y_test) * 100,
'Intercept' : model.intercept_,
'Mean Absolute Error' : metrics.mean_absolute_error(y_test, y_pred),
'Mean Squared Error' : metrics.mean_squared_error(y_test, y_pred)
})


#  Model Metrics
st.subheader('Model Metrics')
st.write("""

Some metric values to give an idea of the model's performance

""")
st.write(metrics)

# PREDICTION SUB HEADER
st.subheader('Predictor')
st.write("""
Model Input
""")
st.write(input_df)

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_df)

    st.write(pd.Series({'Prediction' : prediction[0] }))






