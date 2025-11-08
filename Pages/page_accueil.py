import streamlit as st
from PIL import Image

def app():
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=600)
    except Exception as e:
        st.warning(f"Logo non trouvé : {e}")

    st.markdown("""
    **Bienvenue sur votre application d'analyse statistique automatisée.**  
    **Cliquez sur le bouton ci-dessous pour démarrer votre exploration des données.**
    """)

    # --- Bouton pour aller à la page "Fichier" ---
    if st.button("Démarrer mon analyse"):
        st.session_state.target_page = "Fichier"  # on redirige via clé intermédiaire

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
