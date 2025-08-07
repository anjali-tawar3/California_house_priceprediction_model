import streamlit as st
import pandas as pd
import random 
from sklearn.preprocessing import StandardScaler
import pickle
import time 


#Title 
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')

st.image('https://is1-3.housingcdn.com/4f2250e8/978669a9e786a157b0bbcc87f81380a8/v0/fs/vaibhav_shantam_nest-tarsali-vadodara-vaibhav_infra.jpeg')

st.header('Model of housing prices to predict median house values in California',divider = True)

# st.subheader(''' User Must Enter Given Values To Predict''')

st.sidebar.title('Select House Features 🏡')

st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20220902/pngtree-rising-house-prices-concept-3d-illustration-sale-construction-render-photo-image_47623692.jpg')

#read data 
temp_df = pd.read_csv('california.csv')
random.seed(13)

all_values = []

for i in temp_df[col]:
    min_value,max_value = temp_df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value), random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_values])


with open('House_price_pred_model_ridge.pkl','rb') as f:
    chatgpt = pickle.load(f)
 
price = chatgpt.predict(final_value)[0]


st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

progress_bar = st.progress(0)
placeholder = st.empty()
time.sleep(.5)
placeholder.subheader('Predicting Price!!!!')
place = st.empty()
place.image('https://cdn-icons-gif.flaticon.com/11677//11677497.gif',width = 80)


if price >0:
    
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
    body = f'Predicted Median House Price : ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
        #st.subheader(body)
    st.success(body)
else :
    body = 'Invalid House Features Values'
    st.warning(body)



