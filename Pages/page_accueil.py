import streamlit as st
from PIL import Image

def app():
    # --- 1️⃣ Page d'accueil ---
    st.set_page_config(page_title="Corvus Analysics", layout="wide")

    # --- 2️⃣ Logo et titre ---
    try:
        logo = Image.open("assets/logo.png")  
        st.image(logo, width=600)
    except Exception as e:
        st.warning(f"Logo non trouvé : {e}")

    st.markdown("""
    **Bienvenue sur votre application d'analyse statistique automatisée.**  
    **Cliquez sur le bouton ci-dessous pour démarrer votre exploration des données.**
    """)

    # --- 4️⃣ Pied de page ---
    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")





