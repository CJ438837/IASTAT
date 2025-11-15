import streamlit as st
from PIL import Image

def app():
   
    st.markdown("""
    **Une question sur l'utilisation, l'interprétation des résultats ou un bug ?**
    **N'hésitez pas à me contacter par mail à l'adresse suivante : corvus.analysis@outlook.com**
    """)

    # Bouton de redirection
    if st.button("📈 Démarrer mon analyse"):
        st.session_state.target_page = "Fichier"

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
