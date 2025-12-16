import streamlit as st
import pandas as pd
import joblib
import os

# ================================
# Konfigurasi halaman
# ================================
st.set_page_config(
    page_title="Adult Income Prediction",
    layout="centered"
)

st.title("🔍 Prediksi Tingkat Pendapatan")
st.write(
    "Aplikasi ini mendemonstrasikan model klasifikasi "
    "Random Forest dan Decision Tree untuk memprediksi "
    "tingkat pendapatan individu (<=50K atau >50K)."
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
    "Random Forest": "model_rf_adult_income.pkl",
    "Decision Tree": "model_dt_adult_income.pkl"
}

model_path = MODEL_PATHS[model_option]

# ================================
# Load Model
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
DATASET_PATH = "adult.csv"

if not os.path.exists(DATASET_PATH):
    st.error(f"❌ File dataset '{DATASET_PATH}' tidak ditemukan.")
    st.stop()

data = pd.read_csv(DATASET_PATH)

# Bersihkan spasi (penting untuk Adult dataset)
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

st.subheader("📊 Preview Dataset")
st.dataframe(data.head())

st.divider()

# ================================
# Prediksi
# ================================
if st.button("🔮 Prediksi Pendapatan"):
    try:
        # Drop target jika ada
        X_data = data.drop("income", axis=1, errors="ignore")

        predictions = model.predict(X_data)

        result = data.copy()
        result["Prediksi_Income"] = predictions

        st.subheader("📈 Hasil Prediksi")
        st.dataframe(result.head(20))

        st.success("✅ Prediksi berhasil dilakukan!")
        st.write("Jumlah prediksi pendapatan >50K:", (predictions == ">50K").sum())
        st.write("Jumlah prediksi pendapatan ≤50K:", (predictions == "<=50K").sum())

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat melakukan prediksi")
        st.code(str(e))
        st.warning(
            "Pastikan struktur kolom dataset sama "
            "dengan dataset yang digunakan saat training model."
        )

# ================================
# Footer
# ================================
st.divider()
st.caption(
    f"Model Aktif: {model_option} | "
    "Tipe: Klasifikasi Biner | "
    "Topik: Adult Census Income Prediction"
)
