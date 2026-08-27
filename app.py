import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import time
import joblib
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Sentimen End-to-End",
   # page_icon="",
    layout="wide"
)

# --- Fungsi Pembersihan Teks ---
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Inisialisasi Session State ---
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# --- Judul Aplikasi ---
st.title("Analisis Sentimen Ulasan Film (ANN)")
st.subheader("Studi Kasus oleh Kelompok 7 (Versi Interaktif)")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["[ 1 ] 📤 Upload Data", 
                            "[ 2 ] 🛠️ Latih Model", 
                            "[ 3 ] 🚀 Prediksi Sentimen"])

# =====================================================
# TAB 1: UPLOAD DATA
# =====================================================
with tab1:
    st.header("Langkah 1: Upload Dataset Anda")
    st.markdown("Silakan upload file CSV berisi ulasan film dan label sentimen (misal: positive / negative).")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.session_state.df = df_upload
            st.success(f"File '{uploaded_file.name}' berhasil di-upload!")
            st.dataframe(df_upload.head())
            st.info(f"Dataset Anda memiliki {df_upload.shape[0]} baris data.")

            # Reset status model
            st.session_state.model_ready = False
        except Exception as e:
            st.error(f"Error saat membaca file: {e}")
            st.session_state.df = None

# =====================================================
# TAB 2: LATIH MODEL
# =====================================================
with tab2:
    st.header("Langkah 2: Konfigurasi dan Pelatihan Model")

    if st.session_state.df is None:
        st.warning("Harap upload dataset di tab '[ 1 ] Upload Data' terlebih dahulu.")
        st.stop()

    df = st.session_state.df
    is_data_valid = False

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("2.1. Pilih Kolom Data")
        text_column = st.selectbox("Pilih kolom Teks (Review):", options=df.columns)
        label_column = st.selectbox("Pilih kolom Label (Sentiment):", options=df.columns, index=min(1, len(df.columns)-1))

        st.subheader("2.2. Atur Hyperparameter")
        use_sample = st.checkbox("Gunakan sampel data", value=True)
        sample_size = st.number_input("Jumlah sampel:", min_value=100, max_value=df.shape[0], value=min(1000, df.shape[0]), disabled=not use_sample)

        max_features = st.number_input("Max Features (TF-IDF):", min_value=100, value=5000)
        hidden_layer_size = st.number_input("Neuron Hidden Layer:", min_value=10, value=100)
        max_iter_ann = st.number_input("Max Iterasi (ANN):", min_value=50, value=200)

    with col2:
        st.subheader("2.3. Pemeriksaan Data Label")
        try:
            value_counts = df[label_column].value_counts()
            st.dataframe(value_counts)
            if len(value_counts) < 2:
                st.error(f"Kolom label '{label_column}' hanya punya {len(value_counts)} nilai unik.")
                is_data_valid = False
            else:
                st.success(f"Data label valid ({len(value_counts)} kelas ditemukan).")
                is_data_valid = True
        except Exception as e:
            st.error(f"Gagal membaca label: {e}")
            is_data_valid = False

    st.markdown("---")

    st.subheader("2.4. Mulai Pelatihan")
    if st.button("LATIH MODEL SEKARANG", type="primary", use_container_width=True):
        if text_column == label_column:
            st.error("Kolom teks dan label tidak boleh sama.")
        elif not is_data_valid:
            st.error("Data label tidak valid. Periksa distribusi label.")
        else:
            with st.spinner("Melatih model, mohon tunggu..."):
                try:
                    # Sampling
                    if use_sample and sample_size < df.shape[0]:
                        try:
                            df_train, _ = train_test_split(
                                df, train_size=int(sample_size),
                                stratify=df[label_column], random_state=42
                            )
                            st.text(f"Mengambil {len(df_train)} sampel data secara stratified.")
                        except ValueError:
                            df_train = df.sample(n=int(sample_size), random_state=42)
                            st.warning("Stratified sampling gagal. Menggunakan random sampling biasa.")
                    else:
                        df_train = df.copy()

                    # Preprocessing
                    df_train['cleaned_review'] = df_train[text_column].apply(clean_text)
                    le = LabelEncoder()
                    df_train['sentiment_encoded'] = le.fit_transform(df_train[label_column])

                    y = df_train['sentiment_encoded']
                    vectorizer = TfidfVectorizer(max_features=int(max_features), ngram_range=(1, 2))
                    X = vectorizer.fit_transform(df_train['cleaned_review'])

                    # Model
                    ann_model = MLPClassifier(
                        hidden_layer_sizes=(int(hidden_layer_size),),
                        max_iter=int(max_iter_ann),
                        activation='relu',
                        solver='adam',
                        random_state=42,
                        early_stopping=True
                    )
                    ann_model.fit(X, y)

                    # Simpan ke session_state
                    st.session_state.model = ann_model
                    st.session_state.vectorizer = vectorizer
                    st.session_state.label_encoder = le
                    st.session_state.model_ready = True

                    # Simpan juga ke file lokal
                    joblib.dump(ann_model, "trained_ann_model.pkl")
                    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
                    joblib.dump(le, "label_encoder.pkl")

                    st.success("Model berhasil dilatih dan disimpan!")
                    st.info("Pindah ke tab '[ 3 ] Prediksi Sentimen' untuk uji prediksi.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat pelatihan: {e}")
                    st.session_state.model_ready = False

# =====================================================
# TAB 3: PREDIKSI SENTIMEN
# =====================================================
with tab3:
    st.header("Langkah 3: Gunakan Model untuk Prediksi")

    # --- Load model dari file jika belum ada ---
    if not st.session_state.model_ready:
        if all(os.path.exists(f) for f in ["trained_ann_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"]):
            try:
                st.session_state.model = joblib.load("trained_ann_model.pkl")
                st.session_state.vectorizer = joblib.load("tfidf_vectorizer.pkl")
                st.session_state.label_encoder = joblib.load("label_encoder.pkl")
                st.session_state.model_ready = True
                st.success("Model berhasil dimuat dari file lokal! ✅")
            except Exception as e:
                st.error(f"Gagal memuat model dari file: {e}")
                st.stop()
        else:
            st.warning("Harap latih model di tab '[ 2 ] Latih Model' terlebih dahulu.")
            st.stop()

    model = st.session_state.model
    vectorizer = st.session_state.vectorizer
    label_encoder = st.session_state.label_encoder

    st.info("Model siap digunakan untuk memprediksi sentimen.")

    # Input teks
    text_input = st.text_area("Tulis ulasan di sini:", height=150, placeholder="Contoh: The movie was amazing and the actors did a great job!")

    if st.button("Prediksi Sentimen", type="primary"):
        if text_input.strip() == "":
            st.warning("Masukkan teks ulasan terlebih dahulu.")
        else:
            cleaned_input = clean_text(text_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction_code = model.predict(vectorized_input)
            prediction_proba = model.predict_proba(vectorized_input)

            prediction_label = label_encoder.inverse_transform(prediction_code)[0]
            confidence = prediction_proba[0][prediction_code[0]] * 100

            positive_label = label_encoder.classes_[-1]
            if prediction_label == positive_label:
                st.success(f"Prediksi: **Sentimen Positif** 😊👍")
                st.balloons()
            else:
                st.error(f"Prediksi: **Sentimen Negatif** 😞👎")

            st.metric(label="Tingkat Keyakinan Prediksi", value=f"{confidence:.2f}%")
            st.write(f"Label yang dikenali model: `{list(label_encoder.classes_)}`")
