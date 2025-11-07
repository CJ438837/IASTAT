import streamlit as st

# Charger le thème CORVUS
with open("assets/corvus_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Appstats",
    layout="wide"
)

# 🔄 Si la page a été modifiée par un bouton
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# --- Menu latéral ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à :",
    [
        "Accueil",
        "Fichier",
        "Variables",
        "Descriptive",
        "Distribution",
        "Tests bivariés",
        "Tests multivariés"
    ],
    index=[
        "Accueil",
        "Fichier",
        "Variables",
        "Descriptive",
        "Distribution",
        "Tests bivariés",
        "Tests multivariés"
    ].index(st.session_state.page)
)

# Synchronisation pour que le bouton fonctionne
st.session_state.page = page

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
elif page == "Tests bivariés":
    from Pages import page_testsbivaries
    page_testsbivaries.app()
elif page == "Tests multivariés":
    from Pages import page_testsmulti
    page_testsmulti.app()
