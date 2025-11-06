import streamlit as st
from PIL import Image

def app():
    # --- 1️⃣ Page d'accueil ---
    st.set_page_config(page_title="Corvus Analysis", layout="wide")

    # --- 2️⃣ Logo et titre ---
    try:
        # Chemin vers ton logo Corvus (à adapter selon ton projet)
        logo = Image.open("assets/logoc.png")  
        st.image(logo, width=600)
    except Exception as e:
        st.warning(f"Logo non trouvé : {e}")

    st.markdown("""
    Bienvenue sur votre application d'analyse statistique automatisée.  
    Sélectionnez une page ci-dessous pour démarrer votre exploration des données.
    """)

    # --- 3️⃣ Boutons de navigation ---
    st.subheader("Navigation rapide")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Analyse Descriptive"):
            st.session_state.current_page = "descriptive"

    with col2:
        if st.button("Analyse de Distribution"):
            st.session_state.current_page = "distribution"

    with col3:
        if st.button("Tests Multivariés"):
            st.session_state.current_page = "multivariee"

    # --- 4️⃣ Optionnel : ajout d'un pied de page ---
    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")




