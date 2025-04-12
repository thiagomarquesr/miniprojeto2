import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = "models/audio_emotion_model.keras"  # Example
SCALER_PATH = "models/scaler.pkl"                # Example
encoder = joblib.load("models/encoder.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    features.extend(zcr)

    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    features.extend(chroma)

    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    features.extend(mfccs)

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features.extend(rms)

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    features.extend(mel)

    target_length = 182
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
st.set_page_config(page_title="Detector de Emoções por Áudio", layout="centered")


st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans&display=swap" rel="stylesheet">
    <style>
        /* Fonte global */
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
        }

        /* Fundo geral da aplicação */
        .stApp {
            background-color: #F8F9FA;
        }

        /* Barra lateral */
        [data-testid="stSidebar"] {
            background-color: #F1F3F5;
        }

        /* Cabeçalho */
        header {
            background-color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Menu", ["Pagina Inicial", "Sobre"], 
        icons=['house', 'info-circle'], menu_icon="cast", default_index=0)


if selected == "Pagina Inicial":

    st.title("Detector de Emoções por Áudio")
    st.write("Envie um arquivo de áudio para que o modelo identifique a emoção presente na fala.")
    
    uploaded_file = st.file_uploader(
        "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        st.audio("temp_audio.wav", format="audio/wav")

        features = extract_features("temp_audio.wav")

        features_scaled = scaler.transform(features)

        features_reshaped = np.expand_dims(features_scaled, axis=2)

        prediction = model.predict(features_reshaped)
        predicted_label = encoder.inverse_transform(prediction)[0]

        st.subheader(f"Emoção detectada: {predicted_label}")

        st.bar_chart(prediction[0])

        os.remove("temp_audio.wav")
    
elif selected == "Sobre":
    st.title("Sobre o Projeto")
    st.markdown("""
    Este projeto foi desenvolvido por Thiago Mateus Marques Ribeiro, estudante do curso de Ciência de Dados e Inteligência Artificial da Universidade Federal da Paraíba (UFPB). Ele faz parte de uma das atividades propostas pela Trilha UFPB, voltada ao desenvolvimento prático de soluções utilizando inteligência artificial.

A ideia central do projeto é identificar emoções humanas a partir de áudios de voz. Para isso, o sistema foi construído em etapas bem definidas. Primeiro, houve a preparação dos dados, com extração de características relevantes dos áudios, como frequência, ritmo e energia. Em seguida, os dados foram normalizados e utilizados para treinar uma rede neural convolucional, capaz de aprender padrões associados a diferentes emoções.

Todo o processo de modelagem teve como objetivo alcançar a maior precisão possível na identificação das emoções. Depois de treinado, o modelo foi integrado a uma aplicação web interativa, construída com Streamlit. Assim, qualquer pessoa pode fazer upload de um arquivo de áudio e ver, em tempo real, a emoção que o modelo identifica.

Este projeto é um exemplo de como a inteligência artificial pode ser aplicada para entender aspectos sutis da comunicação humana, como as emoções transmitidas pela voz.
    """)