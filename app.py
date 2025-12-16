import streamlit as st
import pandas as pd
import joblib
import os
import sklearn # Import untuk cek versi jika error

# ================================
# 1. Konfigurasi Halaman
# ================================
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("🔍 Prediksi Tingkat Pendapatan")
st.write(
    "Aplikasi ini mendemonstrasikan model klasifikasi "
    "untuk memprediksi tingkat pendapatan individu (<=50K atau >50K)."
)
st.divider()

# ================================
# 2. Fungsi Load Model (Dengan Cache)
# ================================
# @st.cache_resource mencegah load ulang model setiap kali user berinteraksi
@st.cache_resource 
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return "ERROR"

# ================================
# 3. Sidebar / Pilihan Model
# ================================
st.subheader("⚙️ Konfigurasi Model")

model_option = st.radio(
    "Pilih Algoritma:",
    ("Random Forest", "Decision Tree"),
    horizontal=True
)

# PENTING: Pastikan file fisik di foldermu bernama 'model_rf_fixed.pkl'
# (File hasil kompresi/rename, BUKAN file lama yang error LFS)
MODEL_PATHS = {
    "Random Forest": "model_rf_fixed.pkl", 
    "Decision Tree": "model_dt_adult_income.pkl"
}

selected_path = MODEL_PATHS[model_option]

# Load model menggunakan fungsi yang sudah di-cache
model = load_model(selected_path)

if model is None:
    st.error(f"❌ File '{selected_path}' tidak ditemukan. Pastikan file sudah di-upload ke GitHub.")
    st.stop()
elif model == "ERROR":
    st.error("❌ Terjadi kesalahan versi library. Cek requirements.txt.")
    st.stop()
else:
    st.success(f"✅ Model {model_option} siap digunakan.")

st.divider()

# ================================
# 4. Load & Preview Dataset
# ================================
DATASET_PATH = "adult.csv"

if os.path.exists(DATASET_PATH):
    data = pd.read_csv(DATASET_PATH)
    
    # Membersihkan spasi di data string (Penting!)
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()
        
    st.subheader("📊 Sampel Data (adult.csv)")
    st.dataframe(data.head())
else:
    st.warning(f"⚠️ File '{DATASET_PATH}' tidak ditemukan. Upload file csv untuk tes prediksi.")

# ================================
# 5. Eksekusi Prediksi
# ================================
st.divider()
if st.button("🔮 Mulai Prediksi (Batch)"):
    if 'data' not in locals():
        st.error("Data tidak tersedia untuk diprediksi.")
    else:
        try:
            with st.spinner('Sedang memproses prediksi...'):
                # Siapkan data X (Fitur)
                # Pastikan kolom target 'income' dibuang sebelum prediksi
                X_data = data.drop("income", axis=1, errors="ignore")

                # Lakukan Prediksi
                predictions = model.predict(X_data)

                # Gabungkan hasil
                result = data.copy()
                result["Hasil_Prediksi"] = predictions

                # Tampilkan Hasil
                st.subheader("📈 Hasil Prediksi")
                st.dataframe(result[["age", "occupation", "education", "Hasil_Prediksi"]].head(20))

                # Statistik Singkat
                col1, col2 = st.columns(2)
                count_gt_50 = (predictions == ">50K").sum()
                count_le_50 = (predictions == "<=50K").sum()

                with col1:
                    st.metric("Prediksi >50K", f"{count_gt_50} Orang")
                with col2:
                    st.metric("Prediksi <=50K", f"{count_le_50} Orang")

        except ValueError as ve:
            st.error("❌ Error Format Data:")
            st.write("Model mengharapkan format input tertentu. Pastikan 'adult.csv' memiliki kolom yang sama persis dengan data saat training.")
            st.expander("Lihat Detail Error").write(ve)
            
        except Exception as e:
            st.error("❌ Terjadi Kesalahan Tak Terduga:")
            st.code(str(e))
            st.info(f"Versi Scikit-Learn di App ini: {sklearn.__version__}")
            st.write("Tips: Pastikan versi scikit-learn di `requirements.txt` sama dengan di laptopmu.")

# ================================
# Footer
# ================================
st.divider()
st.caption("Tugas Akhir Data Mining - Prediksi Pendapatan")