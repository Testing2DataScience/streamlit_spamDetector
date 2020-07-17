import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import pickle
from PIL import Image



trans_pkl = open("transform.pkl","rb")
prod_transf = joblib.load(trans_pkl)

clf_vect_tfidf= open("pickle.pkl","rb")
prod_clf = joblib.load(clf_vect_tfidf)


def main():
    st.title("SPAM DETECTOR................")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit  ML App </h2>
    </div>
    """
    user_input = st.text_input("Enter text ", )   
    
    
    if (st.button ("Predict")):

        vect = prod_transf.transform([user_input])
        my_pred = prod_clf.predict(vect)

        if my_pred == 1 :    
            st.write('This is not  spam !!!!!!!!!!')
        elif my_pred ==0:
            st.write('BEWARE This is spam')
    


if __name__ == '__main__':
    main()


