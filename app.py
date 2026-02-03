import io
import pandas as pd
import streamlit as st
import os
import re
import emoji
import string
import time
import subprocess

from bs4 import BeautifulSoup
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv
from supabase import create_client, Client

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sentimen MBG IndoBERT", layout="wide", page_icon="🛡️")
load_dotenv()

st.markdown("""
    <style>
    [data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        padding-left: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- KONFIGURASI SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL_HF_REPO = "finoraf/sentimen-mbg-model"

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Konfigurasi Supabase (.env) tidak ditemukan!")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- INIT NLP & MODEL (CACHED) ---
@st.cache_resource
def load_resources():
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_HF_REPO)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return stopword_remover, stemmer, pipe

stopword_remover, stemmer, pipe = load_resources()

label_map = {'label_0': 'Negatif', 'label_1': 'Netral', 'label_2': 'Positif'}

# --- SESSION STATE ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_email' not in st.session_state: st.session_state['user_email'] = None
if 'active_menu' not in st.session_state: st.session_state['active_menu'] = "dashboard"

# --- FUNGSI HELPER ---
@st.cache_data
def load_slang_dict():
    try:
        df = pd.read_csv('colloquial-indonesian-lexicon.csv')
        return dict(zip(df['slang'], df['formal']))
    except:
        return {}

slang_dict = load_slang_dict()

def clean_text(text):
    text = str(text).lower()
    words = text.split()
    text = ' '.join([slang_dict.get(word, word) for word in words])
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www.\S+|@\w+|#\w+", "", text)
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"[^\x00-\x7F]+|\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    return stemmer.stem(text)

def save_prediction(teks_asli, teks_bersih, label, skor):
    try:
        data = {
            "user_email": st.session_state['user_email'],
            "teks_asli": teks_asli,
            "teks_bersih": teks_bersih,
            "sentimen": label,
            "skor": skor
        }
        supabase.table("history_klasifikasi").insert(data).execute()
    except Exception as e:
        st.error(f"Gagal simpan ke database: {e}")

# --- CONTENT LOGIC ---
if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
    
    with col_tengah:
        st.markdown("<h1 style='text-align: center;'>Aplikasi Analisis Sentimen indoBERT</h1>", unsafe_allow_html=True)
        st.info("Silakan masuk untuk mengakses dashboard analisis.")
        
        with st.form("login_form_main"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Masuk Sekarang", use_container_width=True)
            
            if submitted:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = res.user.email
                    st.success("Login berhasil")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal Login: {e}")

else:

    with st.sidebar:
        st.title("Aplikasi Analisis Sentimen IndoBERT")
        st.success(f"User: {st.session_state['user_email']}")
        
        if st.button("Dashboard", use_container_width=True):
            st.session_state['active_menu'] = "dashboard"
            st.rerun()

        if st.button("Scraping Data", use_container_width=True):
            st.session_state['active_menu'] = "scraping"
            st.rerun()
            
        if st.button("Dataset & Preprocessing", use_container_width=True):
            st.session_state['active_menu'] = "dataset"
            st.rerun()

        if st.button("Prediksi", use_container_width=True):
            st.session_state['active_menu'] = "prediksi"
            st.rerun()
            
        if st.button("History", use_container_width=True):
            st.session_state['active_menu'] = "history"
            st.rerun()
        
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- DASHBOARD ---
    if st.session_state['active_menu'] == "dashboard":
        st.header("📊 Statistik Real-time")
        res = supabase.table("history_klasifikasi").select("sentimen").eq("user_email", st.session_state['user_email']).execute()
        
        if res.data:
            df_dash = pd.DataFrame(res.data)
            df_dash['sentimen_lower'] = df_dash['sentimen'].str.lower()
            counts = df_dash['sentimen_lower'].value_counts()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Data", len(df_dash))
            c2.metric("🟢 Positif", counts.get('positif', 0) + counts.get('positive', 0))
            c3.metric("⚪ Netral", counts.get('netral', 0) + counts.get('neutral', 0))
            c4.metric("🔴 Negatif", counts.get('negatif', 0) + counts.get('negative', 0))
            
            st.divider()
            st.subheader("Distribusi Sentimen")
            st.bar_chart(counts)
        else:
            st.info("Belum ada data di database. Silakan lakukan prediksi.")

    # --- SCRAPING DATA ---
    elif st.session_state['active_menu'] == "scraping":
        st.header("🕷️ Scraping Data X (Twitter)")
        st.info("Fitur ini menggunakan **Tweet Harvest (Node.js)**. Pastikan Node.js terinstal di sistem ini.")

        with st.form("scraping_form"):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                search_keyword = st.text_input("Kata Kunci Pencarian", value="Makan Bergizi Gratis lang:id")
                limit_tweets = st.number_input("Jumlah Tweet (Limit)", min_value=10, max_value=2000, value=20, step=10)
            with col_s2:
                auth_token = st.text_input("Twitter Auth Token", type="password", help="Hanya masukkan kode uniknya saja.")
                filename_output = st.text_input("Nama File Output (tanpa .csv)", value="hasil_scraping")
            
            submit_scrape = st.form_submit_button("Mulai Scraping 🚀", use_container_width=True)

        if submit_scrape:
            if not auth_token:
                st.warning("Mohon masukkan Auth Token Twitter Anda.")
            else:
                clean_token = auth_token.strip().replace('"', '').replace("'", "")
                if clean_token.startswith("auth_token="):
                    clean_token = clean_token.replace("auth_token=", "")
                
                output_file = f"{filename_output}.csv"
                
                command = [
                    "npx", "-y", "tweet-harvest@latest",
                    "-o", output_file,
                    "-s", search_keyword,
                    "-l", str(limit_tweets),
                    "--token", clean_token
                ]

                try:
                    with st.spinner(f"Sedang mencari tweet tentang '{search_keyword}'..."):
                        process = subprocess.run(command, capture_output=True, text=True, shell=True)

                    if process.returncode == 0:
                        folder_path = os.path.join(os.getcwd(), "tweets-data", output_file)
                        root_path = output_file 
                        
                        final_path = None
                        if os.path.exists(folder_path):
                            final_path = folder_path
                        elif os.path.exists(root_path):
                            final_path = root_path

                        if final_path:
                            st.success(f"✅ Scraping Selesai! {limit_tweets} tweet berhasil diambil.")
                            
                            df_scraped = pd.read_csv(final_path)
                            st.write(f"**Preview Data ({len(df_scraped)} baris):**")
                            st.dataframe(df_scraped.head())

                            csv_data = df_scraped.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Hasil Scraping", csv_data, output_file, "text/csv")
                            st.info("Download file ini, lalu upload di menu **📤 Dataset & Preprocessing**.")
                        else:
                            st.warning(f"Proses selesai, tapi file tidak ditemukan di folder 'tweets-data'.")
                            st.caption(f"Sistem mencari di: {folder_path}")
                            with st.expander("Lihat Log Proses"):
                                st.code(process.stdout)
                    else:
                        st.error("Terjadi kesalahan sistem (Token Invalid atau Koneksi).")
                        with st.expander("Lihat Detail Error"):
                            st.code(process.stderr)

                except Exception as e:
                    st.error(f"Gagal menjalankan perintah: {e}")

    # --- DATASET & PREPROCESSING ---
    elif st.session_state['active_menu'] == "dataset":
        st.header("📤 Manajemen Dataset")
        up_file = st.file_uploader("Upload file CSV", type=["csv"])
        if up_file:
            df = pd.read_csv(up_file, sep=None, engine='python')
            st.write("Preview Data Asli:")
            st.dataframe(df.head())
            
            target_col = st.selectbox("Pilih Kolom Teks Opini:", df.columns)
            if st.button("Proses Preprocessing"):
                with st.spinner("Membersihkan data..."):
                    df['teks_bersih'] = df[target_col].apply(clean_text)
                    st.session_state['data_ready'] = df
                    st.session_state['col_asli'] = target_col
                    st.success("Preprocessing Selesai!")
                    st.dataframe(df[[target_col, 'teks_bersih']].head())

    # --- PREDIKSI ---
    elif st.session_state['active_menu'] == "prediksi":
        st.header("🤖 Klasifikasi Sentimen")
        mode = st.radio("Pilih Mode:", ["Teks Tunggal", "Dataset Massal"])
        
        if mode == "Teks Tunggal":
            input_txt = st.text_area("Masukkan kalimat opini:")
            if st.button("Analisis Teks"):
                if input_txt:
                    cln = clean_text(input_txt)
                    out = pipe(cln)[0]
                    lbl = label_map.get(out['label'], out['label'])
                    save_prediction(input_txt, cln, lbl, round(out['score']*100, 2))
                    st.success(f"Hasil: **{lbl}** ({round(out['score']*100, 2)}%)")
        
        elif mode == "Dataset Massal":
                if 'data_ready' in st.session_state:
                    df_massal = st.session_state['data_ready']
                    col_txt = st.session_state['col_asli']
                    
                    st.write(f"Siap memproses {len(df_massal)} baris.")
                    
                    if st.button("Mulai Prediksi Massal"):
                        prog_text = "Memulai proses..."
                        prog = st.progress(0, text=prog_text)
                        total_rows = len(df_massal)
                        
                        for i, row in df_massal.iterrows():
                            teks_input = str(row['teks_bersih']) 
                            current_progress = (i + 1) / total_rows
                            current_percent = int(current_progress * 100)

                            if not teks_input.strip() or teks_input.lower() == 'nan':
                                prog.progress(current_progress, text=f"Sedang memproses... {current_percent}%")
                                continue
                                
                            try:
                                pred = pipe(teks_input)[0]
                                res_lbl = label_map.get(pred['label'], pred['label'])
                                
                                save_prediction(str(row[col_txt]), teks_input, res_lbl, round(pred['score']*100, 2))
                                
                            except Exception as e:
                                st.error(f"Error pada baris {i}: {e}")
                            
                            prog.progress(current_progress, text=f"Sedang memproses... {current_percent}%")
                        
                        st.success("Semua data berhasil diproses dan disimpan!")
                else:
                    st.warning("Silakan upload dan proses data di menu Dataset terlebih dahulu.")

    # --- HISTORY ---
    elif st.session_state['active_menu'] == "history":
        st.header("📜 Riwayat Klasifikasi")
        res_hist = supabase.table("history_klasifikasi").select("*").eq("user_email", st.session_state['user_email']).order("created_at", desc=True).execute()
        
        if res_hist.data:
            df_hist = pd.DataFrame(res_hist.data)
            st.dataframe(df_hist[['created_at', 'teks_bersih', 'sentimen', 'skor']], use_container_width=True)
            
            st.divider()
            c_h1, c_h2 = st.columns(2)
            
            with c_h1:
                st.subheader("🗑️ Hapus Baris")
                to_del = st.selectbox("Pilih Teks untuk Dihapus:", df_hist['teks_bersih'].tolist(), index=None)
                if st.button("Hapus Data Terpilih"):
                    if to_del:
                        id_del = df_hist[df_hist['teks_bersih'] == to_del]['id'].values[0]
                        supabase.table("history_klasifikasi").delete().eq("id", id_del).execute()
                        st.rerun()
            
            with c_h2:
                st.subheader("🔥 Reset Data")
                if st.button("Hapus Semua Riwayat", type="primary"):
                    supabase.table("history_klasifikasi").delete().eq("user_email", st.session_state['user_email']).execute()
                    st.rerun()
            
            csv = df_hist[['created_at', 'teks_bersih', 'sentimen', 'skor']].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil (CSV)", data=csv, file_name="hasil_sentimen_mbg.csv", mime="text/csv")
        else:
            st.info("Riwayat kosong.")