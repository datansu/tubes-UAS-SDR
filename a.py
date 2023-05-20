import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np 
import pandas as pd 
from PIL import Image
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


df = pd.read_csv("Admission_Predict.csv") 


df.drop(columns='Serial No.',inplace = True)

df["Admit"] = np.where(df['Chance of Admit '] <= 0.5, 0, 1)

df.columns = df.columns.str.replace(' ', '')

def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    Percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, Percentage], axis=1, keys=['Total', 'Percentage'])
missing_data(df)


X= df.drop(["ChanceofAdmit", "Admit"],axis =1)
y= df["Admit"]

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=7)

xgb = XGBClassifier()

xgb.fit(X_train,y_train)

# web title
st.set_page_config(
    page_title="Ini Judul Halaman",
)

# navigation/option
with st.sidebar:
   selected = option_menu(
        menu_title="Menu Utama",  
        options=["Beranda", "Prediksi"], 
        icons=["house", "record-circle"],  
        menu_icon="cast",  # optional
        default_index=0,  # optional         
)

if selected == "Beranda":
    st.write("# Ini Judul di Beranda")
   

    #image1 = Image.open('download.jpeg')
    
    #st.image(image1)


if selected == "Prediksi":
    st.title("Ini Judul di Prediksi")
    st.write("Tambahkan kalimat disini kalau diperlukan")

    
    GREScore = st.number_input("GRE Score")
    TOEFLScore = st.number_input("TOEFL Score")
    UniversityRating = st.number_input("University Rating")
    SOP = st.number_input("SOP")
    LOR = st.number_input("LOR")
    CGPA = st.number_input("CGPA")
    Research = st.number_input("Research")
    
   
    
    ok = st.button ("Prediksi")

    if ok:
      x_new = [[GREScore, TOEFLScore, UniversityRating, SOP, LOR, CGPA,
       Research]]
      result = xgb.predict(x_new)
      if result == 0:
        st.subheader("No")
      if result == 1:
        st.subheader("Yes")
