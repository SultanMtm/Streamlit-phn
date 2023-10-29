import pickle
import streamlit as st

model = pickle.load(open('Estimasi_Phone_Price.sav', 'rb'))

st.title('Estimasi Harga Hp')
weight = st.number_input('Masukan Tinggi Hp (mm)',)
Sale = st.number_input('Total Penjualan Hp', min_value = 1, step = 1)
cpu_core = st.number_input('Masukan CPU core Hp', min_value = 0, step = 1)
internal_mem = st.number_input('Masukan Total Internal Memori', min_value = 0, step = 1, max_value= 256)
ram = st.number_input('Masukan Total RAM', min_value = 0, step =1, max_value= 8)
RearCam = st.number_input('Masukan Pixel Camera Belakang')
Front_Cam = st.number_input('Masukan Pixel Camera Depan')
battery = st.number_input('Masukan Total Baterai',min_value = 0, step = 10,max_value=10000)
thickness = st.number_input('Masukan Ketebalan Hp (mm)')

predict = ''

if st.button('Estimasi Harga'):
    predict = model.predict(
        [[weight, Sale, cpu_core, internal_mem, ram, RearCam, Front_Cam, battery, thickness]]
    )
    st.write ('Estimasi Harga Hp dalam Riyal: ', predict)