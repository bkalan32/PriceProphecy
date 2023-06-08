import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import matplotlib
from IPython import get_ipython
from PIL import Image


# load the encoder and model object
model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib")
st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(page_title="House Price Prediction App",
        page_icon="", layout="wide")

#creating option list for dropdown menu
options_Bathrooms = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
options_Bedrooms = [1,2,3,4,5,6,7,8]
options_State = ["AL","AK","AZ","AR",'CA','CO','CT','DE','FL','GA','HI','ID',
                'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS',
                'MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
                'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV',
                'WI','WY']

# features list
features = ["price","region","baths","beds","sqft","lot_size","year_built"]


# Give a title to web app using html syntax
st.markdown("<h1 style='text-align: center;'>Price Prophecy Prediction App</h1>", unsafe_allow_html=True)

# define a main() function to take inputs from user in form based approach
def main():
        with st.form("Home_Price_prediction_form"):
          st.subheader("Please enter the following inputs:")

          year_built = st.slider("how old is your home",1970,2023, value=1970, format="%d")
          square_ft = st.slider("How big is your home in Sqrft",1,5000, value=0, format="%d")
          bedrooms = st.selectbox("Number of bedrooms", options_Bedrooms)
          bathrooms = st.selectbox("Number of bathrooms", options=options_Bathrooms)
          Acre_lot = st.number_input("Whats the size of the lot",min_value=0.01, 
                                                                max_value=35.0, 
                                                                value=0, step=0.01,
                                                                format="%.2f", key="Acre_lot" )
          
          submitted = st.form_submit_button("Submit")
# encode using ordinal encoder and predict
          if submitted:
             input_array = np.array([year_built,
                  square_ft,bedrooms,bathrooms,
                  Acre_lot], ndmin=2)
        
        encoded_arr = list(encoder.transform(input_array).ravel())
        
        num_arr = []
        pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1) 

# predict the target from all the input features
        prediction = model.predict(pred_arr)
        st.write(prediction)

# run the main function        
if __name__ == '__main__':
  main()