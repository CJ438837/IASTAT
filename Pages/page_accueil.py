import streamlit as st
import streamlit as st
from PIL import Image

def app():
    st.title("Bienvenue sur IA_STAT")

    # --- Affichage du logo ---
    logo = Image.open("assets/logo.png")  # chemin relatif à ton fichier .py
    st.image(logo, width=200)  # tu peux ajuster la largeur en px

    st.write("Analyse statistique interactive pour vos données.")


