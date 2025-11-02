import streamlit as st

st.set_page_config(
    page_title="Appstats",
    layout="wide"
)

st.title("📊 Appstats - Analyse Statistique Interactive")

# --- Menu latéral ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", [
    "Accueil",
    "Fichier",
    "Variables",
    "Descriptive",
    "Distribution",
    "Tests"
])

# --- Chargement des pages ---
if page == "Accueil":
    from Pages import page_accueil
    page_accueil.app()
elif page == "Fichier":
    from Pages import page_fichier
    page_fichier.app()
elif page == "Variables":
    from Pages import page_variables
    page_variables.app()
elif page == "Descriptive":
    from Pages import page_descriptive
    page_descriptive.app()
elif page == "Distribution":
    from Pages import page_distribution
    page_distribution.app()
elif page == "Tests":
    from Pages import page_tests
    page_tests.app()



