import streamlit as st

# --- 🔧 Thème CORVUS ---
with open("assets/corvus_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Appstats", layout="wide")

# --- ⚙️ Initialisation ---
if "target_page" not in st.session_state:
    st.session_state.target_page = "Accueil"


# ======================================================
# 🖼️ LOGO (au-dessus de la navbar)
# ======================================================
# Mets ton logo dans /assets/logo.png
st.markdown(
    """
    <div style="text-align:center; margin-top: -30px; margin-bottom: 10px;">
        <img src="assets/logo.png" width="160">
    </div>
    """,
    unsafe_allow_html=True
)


# ======================================================
# 🧭 NAVIGATION HORIZONTALE DOUCE
# ======================================================

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

# --- Styles doux ---
st.markdown("""
<style>
.navbar {
    display: flex;
    gap: 10px;
    padding: 8px 18px;
    background-color: #f3f4f6;
    border-radius: 10px;
    margin-bottom: 25px;
    border: 1px solid #e1e1e1;
}

.navbar-button {
    padding: 7px 14px;
    background-color: #ffffff;
    border-radius: 6px;
    color: #333333;
    text-decoration: none;
    font-weight: 500;
    border: 1px solid #cccccc;
    transition: 0.2s ease-in-out;
}

.navbar-button:hover {
    background-color: #e8f0fe;       /* bleu léger */
    border-color: #b8d4ff;
    color: #1a3f8b;
}

.navbar-active {
    background-color: #dbe8ff;       /* bleu pastel */
    border-color: #a7c5ff;
    color: #1a3f8b;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Construire la navbar ---
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

# --- Synchronisation URL ---
if "page" in st.query_params:
    qp = st.query_params["page"]
    if qp in PAGES:
        st.session_state.target_page = qp


# ======================================================
# 🚀 Chargement dynamique de page
# ======================================================
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
