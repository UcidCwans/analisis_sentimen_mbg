import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# MODEL_DIR = "./saved_model"
MODEL_HF_REPO = "rafino/sentimen-mbg-model"

@st.cache_resource
def load_model_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF_REPO)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe

pipe = load_model_pipeline()

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
    result = pipe(text)[0]
    label = label_map.get(result['label'].upper(), result['label'])
    st.write(f"**Sentimen: {label}** ({round(result['score']*100, 2)}%)")