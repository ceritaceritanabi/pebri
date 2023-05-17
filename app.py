import streamlit
import pycaret
import pandas as pd

# Importing the df
df = pd.read_csv('y2.csv')
x = df.iloc[:, 0:2].values
y = df.iloc[:, -1].values
df.head()


from pycaret.regression import *
s = setup(df, target = 'y2', session_id=123)


best = compare_models()


omp_model = create_model('omp')
evaluate_model(omp_model)
predict_model(omp_model)

save_model(omp_model, model_name = 'Orthogonal Matching Pursuit')

from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


    
model = load_model('Orthogonal Matching Pursuit')

def predict(model, df):
    
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['label'[0]
  
                                
def run():


st.title('Thermal Deformation Prediction')
st.write('This is a web app to predict the thermal deformation based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction.')

C_axis_rear_bearing = st.sidebar.slider(label = 'C-axis_rear_bearing', min_value = 0.0,
                          max_value = 100.00 ,
                          value = 20.0,
                          step = 0.1)

A_axis_rear_bearing = st.sidebar.slider(label = 'A-axis rear bearing', min_value = 0.0,
                          max_value = 100.00 ,
                          value = 20.0,
                          step = 1.0)

features = {'C-axis rear bearing': C_axis_rear_bearing, 'A-axis rear bearing': A_axis_rear_bearing
        
            }
 

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict(model=model, features_df=features=df)
    prediction = '$' + (prediction)
    
    st.success(' Based on feature values, the thermal deformation is {}'.format(prediction))
