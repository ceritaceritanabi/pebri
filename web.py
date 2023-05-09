import pandas as pd
import numpy as np
# Importing the dataset
dataset = pd.read_csv('y2.csv')
x = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, -1].values
dataset.head()


import pycaret
from pycaret.regression import *
s = setup(dataset, target = 'y2', session_id=123)


best = compare_models()

exp_clf102 = setup(dataset, target = 'y2',
     session_id=123, normalize = True, transformation = True)    


best = compare_models()


omp_model = create_model('omp')


evaluate_model(omp_model)


predict_model(omp_model)



save_model(omp_model, model_name = 'Orthogonal Matching Pursuit')



from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, dataset):
    
    predictions_data = predict_model(estimator = model, data = dataset)
    return predictions_data['Label'][0]
    
model = load_model('Orthogonal Matching Pursuit')


st.title('Wine Quality Classifier Web App')
st.write('This is a web app to classify the quality of your wine based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the classifier.')




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
    
    prediction = predict_quality(model, features_df)
    
    st.write(' Based on feature values, your wine quality is '+ str(prediction))


