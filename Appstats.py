import streamlit as st

# --- 🔧 Thème CORVUS ---
with open("assets/corvus_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Appstats",
    layout="wide"
)

# --- ⚙️ Initialisation ---
if "page" not in st.session_state:
    st.session_state.page = "Accueil"
if "target_page" not in st.session_state:
    st.session_state.target_page = st.session_state.page

# --- 🧭 Menu latéral ---
st.sidebar.title("Navigation")

pages = [
    "Accueil",
    "Fichier",
    "Variables",
    "Descriptive",
    "Distribution",
    "Tests bivariés",
    "Tests multivariés"
]

# Radio synchronisé avec la page actuelle
page = st.sidebar.radio(
    "Aller à :",
    pages,
    index=pages.index(st.session_state.page),
    key="page"
)

# Synchronisation : si une redirection est demandée depuis une autre page
if st.session_state.page != st.session_state.target_page:
    st.session_state.page = st.session_state.target_page

# --- 🚀 Chargement des pages ---
if st.session_state.page == "Accueil":
    from Pages import page_accueil
    page_accueil.app()
elif st.session_state.page == "Fichier":
    from Pages import page_fichier
    page_fichier.app()
elif st.session_state.page == "Variables":
    from Pages import page_variables
    page_variables.app()
elif st.session_state.page == "Descriptive":
    from Pages import page_descriptive
    page_descriptive.app()
elif st.session_state.page == "Distribution":
    from Pages import page_distribution
    page_distribution.app()
elif st.session_state.page == "Tests bivariés":
    from Pages import page_testsbivaries
    page_testsbivaries.app()
elif st.session_state.page == "Tests multivariés":
    from Pages import page_testsmulti
    page_testsmulti.app()
