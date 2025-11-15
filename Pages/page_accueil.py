import streamlit as st
from PIL import Image

def app():
    st.markdown("""
    **Bienvenue sur votre application d'analyse statistique automatisée.**  
    **Cliquez sur le bouton ci-dessous pour démarrer votre exploration des données.**
    """)

    # Bouton de redirection vers Analyse → Fichier
    if st.button("📈 Démarrer mon analyse"):
        # Définir la page principale sur "Analyse"
        st.session_state.main_page = "Analyse"
        # Définir la sous-page sur "Fichier"
        st.session_state.analyse_subpage = "Fichier"

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
