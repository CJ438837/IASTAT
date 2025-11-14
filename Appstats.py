import streamlit as st

# --- 🔧 Thème CORVUS ---
with open("assets/corvus_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Appstats", layout="wide")

# --- ⚙️ Initialisation ---
if "target_page" not in st.session_state:
    st.session_state.target_page = "Accueil"

# ======================================
# 🧭 NAVIGATION HORIZONTALE EN BANDEAU
# ======================================

PAGES = [
    "Accueil",
    "Fichier",
    "Variables",
    "Descriptive",
    "Distribution",
    "Tests bivariés",
    "Tests multivariés",
    "Contact"
]

# --- Barre horizontale personnalisée ---
st.markdown("""
<style>
.navbar {
    display: flex;
    gap: 12px;
    padding: 10px 20px;
    background-color: #1f1f1f;
    border-radius: 8px;
    margin-bottom: 25px;
}
.navbar-button {
    padding: 8px 16px;
    background-color: #333;
    border-radius: 6px;
    color: white;
    text-decoration: none;
    font-weight: 500;
    border: 1px solid #444;
}
.navbar-button:hover {
    background-color: #555;
}
.navbar-active {
    background-color: #4c8bf5 !important;
    border-color: #4c8bf5 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Construction dynamique du menu ---
nav_html = '<div class="navbar">'
for page in PAGES:
    css_class = "navbar-button"
    if st.session_state.target_page == page:
        css_class += " navbar-active"

    nav_html += (
        f"<a class='{css_class}' href='?page={page.replace(' ', '%20')}'>{page}</a>"
    )
nav_html += "</div>"

st.markdown(nav_html, unsafe_allow_html=True)

# --- Synchronisation avec l’URL ---
if "page" in st.query_params:
    qp = st.query_params["page"]
    if qp in PAGES:
        st.session_state.target_page = qp

# ======================================
# 🚀 Chargement dynamique de la page
# ======================================
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
