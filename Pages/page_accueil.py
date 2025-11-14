import streamlit as st
from PIL import Image

def app():
    
    st.markdown("""
    **Bienvenue sur votre application d'analyse statistique automatisée.**  
    **Cliquez sur le bouton ci-dessous pour démarrer votre exploration des données.**
    """)

    # Bouton de redirection
    if st.button("📈 Démarrer mon analyse"):
        st.session_state.target_page = "Fichier"

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")


