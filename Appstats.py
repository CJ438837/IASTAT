import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Appstats", layout="wide")

# --- INIT ---
if "target_page" not in st.session_state:
    st.session_state.target_page = "Accueil"

# ==========================================================
# 🖼️ LOGO (version Streamlit → fonctionne vraiment)
# ==========================================================
st.markdown("<div style='text-align:center; margin-top:-20px;'>", unsafe_allow_html=True)
st.image("assets/logo.png", width=160)
st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# 🧭 NAVIGATION HORIZONTALE (Streamlit native, pas d'ouverture de page)
# ==========================================================

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

# Style doux
st.markdown("""
<style>
.navbar-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    background-color: #f4f5f7;
    padding: 10px 20px;
    border-radius: 10px;
    margin-top: 10px;
    border: 1px solid #dcdcdc;
}

.nav-btn {
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid #cfcfcf;
    background-color: white;
    color: #333;
    font-weight: 500;
    cursor: pointer;
    transition: 0.2s;
}

.nav-btn:hover {
    background-color: #e8f0fe;
    border-color: #b8d4ff;
    color: #1a3f8b;
}

.nav-btn-active {
    background-color: #dbe8ff;
    border-color: #a7c5ff;
    color: #1a3f8b;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Construction de la navbar
nav_html = "<div class='navbar-container'>"
for page in PAGES:
    active = "nav-btn-active" if st.session_state.target_page == page else ""
    nav_html += f"""
        <button class='nav-btn {active}' onclick="window.location.href='/?nav={page}'">
            {page}
        </button>
    """
nav_html += "</div>"

st.markdown(nav_html, unsafe_allow_html=True)

# Lecture du paramètre de navigation
if "nav" in st.query_params:
    page = st.query_params["nav"]
    if page in PAGES:
        st.session_state.target_page = page
        # On retire le paramètre pour éviter l'effet "reload"
        st.query_params.clear()



# ==========================================================
# 🚀 Chargement dynamique des pages
# ==========================================================
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
