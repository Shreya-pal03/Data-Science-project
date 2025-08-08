import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time


# Title

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://stylesatlife.com/wp-content/uploads/2023/08/Old-Village-House-Design.jpg')



st.header('Model of housing prices to predict median house values in California ',divider=True)

st.subheader('''User Must Enter Given values to predict Price:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')


st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20230804/pngtree-an-upside-graph-showing-prices-and-houses-in-the-market-image_13000262.jpg')


# read_data
temp_df = pd.read_csv('california.csv')

random.seed(12)

all_values = []
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[0]))
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
    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)











