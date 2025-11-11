import streamlit as st

# --- 🔧 Thème CORVUS ---
with open("assets/corvus_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Appstats", layout="wide")

# --- ⚙️ Initialisation ---
if "target_page" not in st.session_state:
    st.session_state.target_page = "Accueil"

# --- 🧭 Menu latéral ---
st.sidebar.title("Navigation")

pages = [
    "Accueil",
    "Fichier",
    "Variables",
    "Descriptive",
    "Distribution",
    "Tests bivariés",
    "Tests multivariés",
    "Contact"
]

# Le radio contrôle la navigation manuelle
selected_page = st.sidebar.radio(
    "Aller à :",
    pages,
    index=pages.index(st.session_state.target_page)
)

# Si on clique dans la sidebar, on met à jour la cible
if selected_page != st.session_state.target_page:
    st.session_state.target_page = selected_page

# --- 🚀 Chargement dynamique de la page ---
if st.session_state.target_page == "Accueil":
    from Pages import page_accueil
    page_accueil.app()
elif st.session_state.target_page == "Fichier":
    from Pages import page_fichier
    page_fichier.app()
elif st.session_state.target_page == "Variables":
    from Pages import page_variables
    page_variables.app()
elif st.session_state.target_page == "Descriptive":
    from Pages import page_descriptive
    page_descriptive.app()
elif st.session_state.target_page == "Distribution":
    from Pages import page_distribution
    page_distribution.app()
elif st.session_state.target_page == "Tests bivariés":
    from Pages import page_testsbivaries
    page_testsbivaries.app()
elif st.session_state.target_page == "Tests multivariés":
    from Pages import page_testsmulti
    page_testsmulti.app()
elif st.session_state.target_page == "Contact":
    from Pages import page_contact
    page_contact.app()



