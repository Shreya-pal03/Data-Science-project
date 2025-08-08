import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title("California House Price Prediction")
st.image("https://media.gettyimages.com/id/183881669/photo/house.jpg?s=612x612&w=gi&k=20&c=vKx9LMH3qNa5n2dLSba8LPjZaSVNwuRDD7B1wisItYU=")
st.write("The project aims at building a model of housing prices to predict median house values in California using the provided dataset. This model should learn from the data and be able to predict the median housing price in any district, given all the other metrics.")

st.header('Model of housing prices to predict median house values in California ',divider=True)

st.subheader('''User Must Enter Given values to predict Price:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')
st.sidebar.title("select house features")
st.sidebar.image("https://media.istockphoto.com/id/1442148484/photo/3d-rendering-of-modern-suburban-house-in-the-garden.jpg?s=612x612&w=0&k=20&c=8Iu_h5cFOEnlPz4_n2nfSUtOyfM_a-hHx9rmlxMF2rI=")
all_values = []
import time
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))
placeholder1 = st.empty()
placeholder1.image("https://cdn.edu.buncee.com/assets/9a210469014b057d19ab922397cf46d8/animation-education-magnifying-glass-042420.gif?timestamp=1587763956",width=50)
temp_df = pd.read_csv('california.csv')
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])
    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))
    all_values.append(var)
ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]

placeholder = st.empty()
placeholder.write("Predicting prices")


if price>0:
    progress_bar = st.progress(0)
    
    for i in range(100):
        time.sleep(0.065)
        progress_bar.progress(i)
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    # st.subheader(body)
    placeholder.empty()
    placeholder1.empty()
    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)

