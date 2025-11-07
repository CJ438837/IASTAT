import streamlit as st
from PIL import Image

def app():
    # --- 1️⃣ Page d'accueil ---
    st.set_page_config(page_title="Corvus Analytics", layout="wide")

    # --- 2️⃣ Logo centré et grande bannière ---
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=600)
    except Exception as e:
        st.warning(f"Logo non trouvé : {e}")

    # --- 3️⃣ Texte d'introduction stylisé ---
    st.markdown("""
    <div style="text-align:center; font-size:22px; font-weight:600; margin-top:20px;">
        Bienvenue sur votre application d'analyse statistique automatisée.<br>
        Cliquez sur le bouton ci-dessous pour démarrer votre exploration des données.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # --- 4️⃣ Bouton principal (vers la page Fichier) ---
    # Fonctionne uniquement si tu as le multipage natif Streamlit
    if st.button("Démarrer mon analyse"):
        st.session_state.page = "Fichier"
        st.session_state.trigger = True  # pas obligatoire mais utile



    # --- 5️⃣ Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#1B2A33; font-size:14px;'>"
        "© 2025 Corvus Analytics - Tous droits réservés"
        "</div>",
        unsafe_allow_html=True,
    )





