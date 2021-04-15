# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:31:08 2021

@author: Anoushka
"""

import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))




def covid_predict(state, cured, confirmed):
    input=np.array([[state, cured, confirmed]]).astype(np.float64)
    prediction = model.predict(input)
    #pred = '{0:.{1}f'.format(prediction[0][0], 2)
    return int(prediction)

def main():
    st.title("Covid-19 Prediction India")
    html_temp = """
    <div style="background-color:4ECECD ;padding:10px">
    <h2 style="color:white;text-align:center;">Covid Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    state = st.text_input("State","Type Here")
    cured = st.text_input("Cured","Type Here")
    confirmed = st.text_input("Confirmed","Type Here")
    
    if st.button("Predict"):
        output=covid_predict(state, cured, confirmed)
        st.success('Deaths: {}'.format(output))

        #if output > 0.5:
            #st.markdown(danger_html,unsafe_allow_html=True)
        #else:
            #st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()