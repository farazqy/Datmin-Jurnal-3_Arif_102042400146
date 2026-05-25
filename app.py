import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Sistem Prediksi Klinis - RS Sehat Sentosa",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_resources():
    try:
        with open('model_linear_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('scaler_linear_regression.pkl', 'rb') as f:
            lr_scaler = pickle.load(f)

        with open('model_naive_bayes_smote.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('scaler_classification.pkl', 'rb') as f:
            nb_scaler = pickle.load(f)

        return lr_model, lr_scaler, nb_model, nb_scaler
    except Exception as e:
        st.error(f"Gagal memuat file model/scaler (.pkl): {e}")
        st.stop()

lr_model, lr_scaler, nb_model, nb_scaler = load_resources()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8254/8254580.png", width=100)
st.sidebar.title("Navigasi Klinis")
halaman = st.sidebar.radio(
    "Pilih Modul Analisis:",
    ["Prediksi Kadar Glukosa (Regresi)", "Klasifikasi Risiko Diabetes (Naive Bayes)"]
)
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini digunakan oleh internal tenaga medis RS Sehat Sentosa untuk menunjang keputusan klinis.")

if halaman == "Prediksi Kadar Glukosa (Regresi)":
    st.title("📈 Estimasi Kadar Glukosa Darah Pasien")
    st.write("Formulir ini memprediksi kadar glukosa darah berdasarkan indikator klinis (Tanpa variabel Outcome & Glucose).")

    with st.form("form_regresi"):
        st.subheader("Input Data Klinis Pasien")
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=40)
            Pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=1)
            BMI = st.number_input("Body Mass Index (BMI * 100)", min_value=1000, value=2500, help="Contoh: Jika BMI 25.0, masukkan 2500")
            BloodPressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=40, value=80)
            HbA1c = st.number_input("Kadar HbA1c (* 10)", min_value=10, value=55, help="Contoh: Jika HbA1c 5.5, masukkan 55")

        with col2:
            LDL = st.number_input("Kadar LDL", min_value=10, value=100)
            HDL = st.number_input("Kadar HDL", min_value=10, value=50)
            Triglycerides = st.number_input("Triglycerides", min_value=10, value=150)
            WaistCircumference = st.number_input("Lingkar Pinggang (Waist)", min_value=30, value=85)
            HipCircumference = st.number_input("Lingkar Pinggul (Hip)", min_value=30, value=95)

        with col3:
            WHR = st.number_input("Waist-Hip Ratio (WHR * 100)", min_value=10, value=89, help="Contoh: Jika WHR 0.89, masukkan 89")
            FamilyHistory = st.selectbox("Riwayat Keluarga Diabetes", options=[0, 1], format_func=lambda x: "Tidak Ada (0)" if x == 0 else "Ada (1)")
            DietType = st.selectbox("Pola Makan (Diet Type)", options=[0, 1], format_func=lambda x: "Sehat (0)" if x == 0 else "Kurang Sehat (1)")
            Hypertension = st.selectbox("Riwayat Hipertensi", options=[0, 1], format_func=lambda x: "Tidak (0)" if x == 1 else "Ya (1)")
            MedicationUse = st.selectbox("Penggunaan Obat-obatan", options=[0, 1], format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)")

        btn_regresi = st.form_submit_button("Hitung Estimasi Glukosa")

    if btn_regresi:
        fitur_regresi = np.array([[
            Age, Pregnancies, BMI, BloodPressure, HbA1c, LDL, HDL,
            Triglycerides, WaistCircumference, HipCircumference, WHR,
            FamilyHistory, DietType, Hypertension, MedicationUse
        ]])

        fitur_scaled = lr_scaler.transform(fitur_regresi)
        hasil_prediksi = lr_model.predict(fitur_scaled)

        st.markdown("---")
        st.metric(label="Estimasi Nilai Glukosa Darah", value=f"{hasil_prediksi[0]:.2f} mg/dL")

elif halaman == "Klasifikasi Risiko Diabetes (Naive Bayes)":
    st.title("🩺 Deteksi Dini Risiko Diabetes Pasien")
    st.write("Formulir ini memprediksi status risiko diabetes (Outcome: 0 atau 1) menggunakan seluruh variabel klinis lengkap.")

    with st.form("form_klasifikasi"):
        st.subheader("Input Rekam Medis Lengkap")
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=45)
            Pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=2)
            BMI = st.number_input("Body Mass Index (BMI * 100)", min_value=1000, value=2800)
            Glucose = st.number_input("Kadar Glukosa Darah (Glucose)", min_value=10, value=140)
            BloodPressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=40, value=85)

        with col2:
            HbA1c = st.number_input("Kadar HbA1c (* 10)", min_value=10, value=60)
            LDL = st.number_input("Kadar LDL", min_value=10, value=120)
            HDL = st.number_input("Kadar HDL", min_value=10, value=45)
            Triglycerides = st.number_input("Triglycerides", min_value=10, value=180)
            WaistCircumference = st.number_input("Lingkar Pinggang (Waist)", min_value=30, value=90)

        with col3:
            HipCircumference = st.number_input("Lingkar Pinggul (Hip)", min_value=30, value=100)
            WHR = st.number_input("Waist-Hip Ratio (WHR * 100)", min_value=10, value=90)
            FamilyHistory = st.selectbox("Riwayat Keluarga Diabetes", options=[0, 1], format_func=lambda x: "Tidak Ada (0)" if x == 0 else "Ada (1)")
            DietType = st.selectbox("Pola Makan (Diet Type)", options=[0, 1], format_func=lambda x: "Sehat (0)" if x == 0 else "Kurang Sehat (1)")
            Hypertension = st.selectbox("Riwayat Hipertensi", options=[0, 1], format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)")
            MedicationUse = st.selectbox("Penggunaan Obat-obatan", options=[0, 1], format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)")

        btn_klasifikasi = st.form_submit_button("Analisis Risiko Diabetes")

    if btn_klasifikasi:
        fitur_klasifikasi = np.array([[
            Age, Pregnancies, BMI, Glucose, BloodPressure, HbA1c, LDL, HDL,
            Triglycerides, WaistCircumference, HipCircumference, WHR,
            FamilyHistory, DietType, Hypertension, MedicationUse
        ]])

        fitur_scaled = nb_scaler.transform(fitur_klasifikasi)
        hasil_klasifikasi = nb_model.predict(fitur_scaled)

        st.markdown("---")
        if hasil_klasifikasi[0] == 1:
            st.error("HASIL ANALISIS: PASIEN BERISIKO TINGGI DIABETES (Kategori: Diabetes)")
            st.markdown("Rekomendasi tindakan medis: Berikan atensi khusus dan jadwalkan uji laboratorium HbA1c berkala.")
        else:
            st.success("HASIL ANALISIS: PASIEN BERISIKO RENDAH (Kategori: Non-Diabetes)**")
            st.markdown("Rekomendasi tindakan medis: Kondisi pasien normal. Edukasikan untuk mempertahankan pola hidup sehat.")
