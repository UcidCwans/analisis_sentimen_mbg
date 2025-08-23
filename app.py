import streamlit as st
import pandas as pd
import re
import emoji
import string

from bs4 import BeautifulSoup
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# MODEL_DIR = "./saved_model"
MODEL_HF_REPO = "finoraf/sentimen-mbg-model"

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = normalize_slang(text, slang_dict)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

def normalize_slang(text, slang_dict):
    """
    Mengganti kata singkatan/slang dalam teks dengan bentuk standarnya
    berdasarkan kamus yang diberikan.
    """
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# --- Fungsi untuk memuat kamus slang dari file CSV ---
@st.cache_data
def load_slang_dictionary(file_path):
    """
    Memuat kamus slang dari file CSV dan mengubahnya menjadi dictionary.
    Menggunakan cache agar file tidak perlu dibaca berulang kali.
    """
    try:
        df_slang = pd.read_csv(file_path)
        # Pastikan nama kolom sesuai dengan file CSV Anda ('slang' dan 'formal')
        return dict(zip(df_slang['slang'], df_slang['formal']))
    except FileNotFoundError:
        st.error(f"File kamus slang tidak ditemukan di '{file_path}'. Pastikan file tersebut ada di direktori yang sama dengan aplikasi Anda.")
        return {} # Kembalikan kamus kosong jika file tidak ada
    
slang_dict = load_slang_dictionary('colloquial-indonesian-lexicon.csv')

@st.cache_resource
def load_model_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF_REPO)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe

# @st.cache_resource
# def load_model_pipeline():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
#     pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
#     return pipe

# pipe = load_model_pipeline()

label_map = {
    'label_0': 'Negatif',
    'label_1': 'Netral',
    'label_2': 'Positif'
}

st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 32px;'>Analisis Sentimen Program Makan Bergizi Gratis Menggunakan Model IndoBERT</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### Hasil Penelitian")
st.markdown("##### Hasil Pelatihan")

st.markdown("o Grafik Akurasi Train")
st.image("train_acc_graphic.png", caption="Akurasi", use_container_width=True)

st.markdown("o Grafik Loss")
st.image("loss_graphic.png", caption="Loss", use_container_width=True)

st.markdown("##### Hasil Pengujian")
st.markdown("o Confusion Matrix")

st.image("cf_matrix.png", caption="Confusion Matrix", use_container_width=True)


st.markdown("#### Prediksi Sentimen")
text = st.text_area("Masukkan Teks:")

if st.button("Analisis"):
    if text:
        text_cleaned = clean_text(text)
        result = pipe(text_cleaned)[0]
        label = label_map.get(result['label'], result['label'])
        score = round(result['score'] * 100, 2)
        
        st.write(f"**Teks yang Dibersihkan:** `{text_cleaned}`")
        st.success(f"**Sentimen: {label}** (Skor: {score}%)")
    else:
        st.warning("Silakan masukkan teks untuk dianalisis.")