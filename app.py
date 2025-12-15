import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="HR Attrition Prediction", layout="centered")

try:
    model = joblib.load("model_hr_attrition.pkl")
except FileNotFoundError:
    st.error("File model 'model_hr_attrition.pkl' tidak ditemukan. Pastikan file tersebut ada di folder yang sama.")
    st.stop()

st.title("🔍 Prediksi Attrition Karyawan")
st.write("Aplikasi ini memprediksi apakah karyawan berpotensi resign atau tidak.")

st.divider()

file_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    
    st.subheader("Preview Data")
    st.dataframe(data.head())

    if st.button("Prediksi Attrition"):
        try:
            # Melakukan prediksi
            prediction = model.predict(data)
            data["Prediksi_Attrition"] = prediction

            st.subheader("Hasil Prediksi")
            st.dataframe(data)

            # Ringkasan
            st.success("Prediksi berhasil dilakukan!")
            st.write("Jumlah karyawan resign:", (prediction == "Yes").sum())
            st.write("Jumlah karyawan tidak resign:", (prediction == "No").sum())
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.warning("Pastikan fitur/kolom dalam dataset sesuai dengan yang digunakan saat training model.")
else:
    st.error(f"File dataset '{file_path}' tidak ditemukan. Mohon pastikan file CSV berada di lokasi yang sama dengan aplikasi ini.")