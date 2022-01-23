import streamlit as st
import sklearn 
# EDA Pkg
import pandas as pd
import joblib
import os
import numpy as np
from PIL import Image

# Load Models
def load_model_n_predict(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)

def remote_css(url):
    st.markdown('<style src="{}"></style>'.format(url), unsafe_allow_html=True)

def icon_css(icone_name):
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# Find the Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key



def main():
	""" ML App with Streamlit"""
	st.title("Who Dies on the Titanic?")
	st.subheader("ML Prediction App with Streamlit")

	# Preview Dataset
	activity = ["prediction"]

	load_css('icon.css')
	# RECEIVE USER INPUT
	#using slider bar
	age = st.slider("Enter Age", 0, 100)
	fare = st.slider("Enter Fare", 0, 600)
	parch = st.slider("Enter Parch", 0, 9)
	pclass = st.slider("Enter Pclass", 1, 3)
	# radio button
	# first argument is the title of the radio button
	# second argument is the options for the ratio button
	sex = st.radio("Select Sex: ", ('Male', 'Female'))
	sibsp = st.slider("Enter Sibsp", 0, 8)
	# USER INPUT ENDS HERE

	# RESULT OF USER INPUT
	#map sex to numeric for model prediction
	for x in sex:
		if sex == 'Male':
			sex = 1
		else:
			sex = 0

	selected_options = [age, fare, parch, pclass, sex, sibsp]
	sample_data = np.array(selected_options).reshape(1, -1)

	# MAKING PREDICTION
	st.subheader("Prediction")
	prediction_label = {"died": 0, "survived": 1}
	if st.button("Predict"):
		#used to be load_model_n_predict("models/stacked_clf.joblib")
		model_predictor = load_model_n_predict("stacked_clf.joblib")
		prediction = model_predictor.predict(sample_data)
		final_result = get_key(prediction,prediction_label)
		st.success(f"This person most likely {final_result}")
				


		#st.video('https://www.youtube.com/watch?v=_9WiB2PDO7k')




if __name__ == '__main__':
	main()
