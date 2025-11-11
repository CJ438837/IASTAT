import streamlit as st
from PIL import Image

def app():
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=600)
    except Exception as e:
        st.warning(f"Logo non trouvé : {e}")

    st.markdown("""
    **Une question sur l'utilisation, l'interprétation des résultats ou un bug ?**
    **N'hésitez pas à me contacter par mail à l'adresse suivante : corvus.analysis@outlook.com**
    """)

    # Bouton de redirection
    if st.button("📈 Démarrer mon analyse"):
        st.session_state.target_page = "Fichier"

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
