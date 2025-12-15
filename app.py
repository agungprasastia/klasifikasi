import streamlit as st
import pandas as pd
import joblib
import os

# ================================
# Konfigurasi halaman
# ================================
st.set_page_config(
    page_title="HR Attrition Prediction",
    layout="centered"
)

st.title("🔍 Prediksi Attrition Karyawan")
st.write(
    "Aplikasi ini mendemonstrasikan model klasifikasi "
    "Random Forest dan Decision Tree untuk memprediksi "
    "attrition karyawan."
)

st.divider()

# ================================
# Pilih Model
# ================================
st.subheader("⚙️ Pilih Model Klasifikasi")

model_option = st.radio(
    "Model yang digunakan:",
    ("Random Forest", "Decision Tree")
)

MODEL_PATHS = {
    "Random Forest": "model_rf_hr_attrition.pkl",
    "Decision Tree": "model_dt_hr_attrition.pkl"
}

model_path = MODEL_PATHS[model_option]

# ================================
# Load model
# ================================
if not os.path.exists(model_path):
    st.error(f"❌ File model '{model_path}' tidak ditemukan.")
    st.stop()

model = joblib.load(model_path)
st.success(f"✅ Model {model_option} berhasil dimuat")

st.divider()

# ================================
# Load Dataset
# ================================
DATASET_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if not os.path.exists(DATASET_PATH):
    st.error(f"❌ File dataset '{DATASET_PATH}' tidak ditemukan.")
    st.stop()

data = pd.read_csv(DATASET_PATH)

st.subheader("📊 Preview Dataset")
st.dataframe(data.head())
st.write(f"Jumlah data: **{data.shape[0]} baris**")

st.divider()

# ================================
# Prediksi
# ================================
if st.button("🔮 Prediksi Attrition"):
    try:
        # Drop target jika ada
        X_data = data.drop("Attrition", axis=1, errors="ignore")

        predictions = model.predict(X_data)

        result = data.copy()
        result["Prediksi_Attrition"] = predictions

        st.subheader("📈 Hasil Prediksi")
        st.dataframe(result.head(20))

        st.success("✅ Prediksi berhasil dilakukan!")
        st.write("Jumlah karyawan diprediksi **resign**:", (predictions == "Yes").sum())
        st.write("Jumlah karyawan diprediksi **tidak resign**:", (predictions == "No").sum())

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat melakukan prediksi")
        st.code(str(e))
        st.warning(
            "Pastikan struktur kolom dataset sama "
            "dengan dataset yang digunakan saat training."
        )

# ================================
# Footer
# ================================
st.divider()
st.caption(
    f"Model Aktif: {model_option} | "
    "Tipe: Klasifikasi Biner | "
    "Topik: Employee Attrition Prediction"
)
