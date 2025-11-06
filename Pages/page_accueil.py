import os
from PIL import Image
import streamlit as st

def app():
    st.title("Bienvenue sur IA_STAT")

    # chemin absolu du fichier logo
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier de page_accueil.py
    logo_path = os.path.join(BASE_DIR, "..", "assets", "logoc.png")
    logo = Image.open(logo_path)
    st.image(logo, width=200)

