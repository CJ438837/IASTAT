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
    "Descriptive",
    "Distribution",
    "Tests"
])

# --- Chargement des pages ---
if page == "Accueil":
    from Pages import 1_Accueil
    1_Accueil.app()
elif page == "Fichier":
    from Pages import 2_fichier
    2_fichier.app()
elif page == "Descriptive":
    from Pages import 3_descriptive
    3_descriptive.app()
elif page == "Distribution":
    from Pages import 4_distribution
    4_distribution.app()
elif page == "Tests":
    from Pages import 5_Tests
    5_Tests.app()
