import numpy as np
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt 
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier

iris_df=pd.read_csv("iris-species.csv")
iris_df['Label']=iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})
X=iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=iris_df['Label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

svc_model=SVC(kernel='linear')
svc_model.fit(X_train,y_train)

rf_clf=RandomForestClassifier(n_estimators=100,n_jobs=-1)
rf_clf.fit(X_train,y_train)

lr=LogisticRegression(n_jobs=-1)
lr.fit(X_train,y_train)

@st.cache()
def prediction(model,SepalLength, SepalWidth, PetalLength, PetalWidth):
  species = model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"

st.title("Iris Flower Species Prediction App")

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_length = st.slider("Sepal Length", 0.0, 10.0)
s_width = st.slider("Sepal Width", 0.0, 10.0)
p_length = st.slider("Petal Length", 0.0, 10.0)
p_width = st.slider("Petal Width", 0.0, 10.0)

classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine','LogisticRegression','RandomForestClassifier'))

if st.sidebar.button("Predict"):
	if classifier=='Support Vector Machine':

		species_type = prediction(svc_model,s_length, s_width, p_length, p_width)
		score=svc_model.score(X_train,y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)

	if classifier=='LogisticRegression':

		species_type = prediction(lr,s_length, s_width, p_length, p_width)
		score=svc_model.score(X_train,y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)

	if classifier=='RandomForestClassifier':

		species_type = prediction(rf_clf,s_length, s_width, p_length, p_width)
		score=svc_model.score(X_train,y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)