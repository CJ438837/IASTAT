import streamlit as st
from PIL import Image

# --- 🔧 Thème CORVUS ---
with open("assets/corvus_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Configuration page ---
st.set_page_config(page_title="Appstats", layout="wide")

# --- ⚙️ Initialisation ---
if "main_page" not in st.session_state:
    st.session_state.main_page = "Accueil"
if "analyse_subpage" not in st.session_state:
    st.session_state.analyse_subpage = "Fichier"
if "theorie_subpage" not in st.session_state:
    st.session_state.theorie_subpage = "Variables"

# ======================================================
# 🖼️ LOGO SUR TOUTE LA LARGEUR
# ======================================================
try:
    logo = Image.open("assets/logo.png")
    st.image(logo, use_container_width=True)
except Exception as e:
    st.warning(f"Logo non trouvé : {e}")

# ======================================================
# 🧭 MENU PRINCIPAL — BANDEAU HORIZONTAL
# ======================================================

MAIN_PAGES = ["Accueil", "Théorie", "Analyse", "Contact"]

st.markdown("""
<style>

.navbar {
    background-color: #0d1117;
    padding: 12px 18px;
    border-radius: 10px;
    margin-top: 10px;
    margin-bottom: 25px;
    display: flex;
    justify-content: center;
    gap: 35px;
    border: 1px solid #30363d;
}

.nav-item {
    color: #e6edf3;
    text-decoration: none;
    font-size: 18px;
    padding: 6px 12px;
    border-radius: 6px;
    transition: 0.2s;
}

.nav-item:hover {
    background-color: #30363d;
}

.nav-active {
    background-color: #238636;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# Création du bandeau
navbar_html = '<div class="navbar">'

for page in MAIN_PAGES:
    active = "nav-active" if st.session_state.main_page == page else ""
    navbar_html += f"""
        <a class="nav-item {active}" href="?page={page}">{page}</a>
    """

navbar_html += "</div>"

st.markdown(navbar_html, unsafe_allow_html=True)

# Mise à jour via URL
query_params = st.query_params
if "page" in query_params:
    st.session_state.main_page = query_params["page"]

# ======================================================
# 📚 LISTES DES SOUS-PAGES
# ======================================================

THEORIE_PAGES = [
    "Variables",
    "Descriptive",
    "Distribution",
    "Tests bivariés",
    "Tests multivariés"
]

ANALYSE_PAGES = [
    "Fichier",
    "Variables",
    "Descriptive",
    "Distribution",
    "Tests bivariés",
    "Tests multivariés"
]

# ======================================================
# 🚀 ROUTEUR DE PAGES
# ======================================================

# --- ACCUEIL ---
if st.session_state.main_page == "Accueil":
    from Pages import page_accueil
    page_accueil.app()

# --- THÉORIE ---
elif st.session_state.main_page == "Théorie":
    st.subheader("Théorie")

    # Sécurité
    if st.session_state.theorie_subpage not in THEORIE_PAGES:
        st.session_state.theorie_subpage = THEORIE_PAGES[0]

    st.session_state.theorie_subpage = st.selectbox(
        "Choisir une section :", THEORIE_PAGES,
        index=THEORIE_PAGES.index(st.session_state.theorie_subpage)
    )

    sub = st.session_state.theorie_subpage

    if sub == "Variables":
        from Pages import page_variablesT; page_variablesT.app()
    elif sub == "Descriptive":
        from Pages import page_descriptiveT; page_descriptiveT.app()
    elif sub == "Distribution":
        from Pages import page_distributionT; page_distributionT.app()
    elif sub == "Tests bivariés":
        from Pages import page_bivariesT; page_bivariesT.app()
    elif sub == "Tests multivariés":
        from Pages import page_multiT; page_multiT.app()

# --- ANALYSE ---
elif st.session_state.main_page == "Analyse":
    st.subheader("Analyse")

    if st.session_state.analyse_subpage not in ANALYSE_PAGES:
        st.session_state.analyse_subpage = ANALYSE_PAGES[0]

    st.session_state.analyse_subpage = st.selectbox(
        "Choisir l'analyse :", ANALYSE_PAGES,
        index=ANALYSE_PAGES.index(st.session_state.analyse_subpage)
    )

    sub = st.session_state.analyse_subpage

    if sub == "Fichier":
        from Pages import page_fichier; page_fichier.app()
    elif sub == "Variables":
        from Pages import page_variables; page_variables.app()
    elif sub == "Descriptive":
        from Pages import page_descriptive; page_descriptive.app()
    elif sub == "Distribution":
        from Pages import page_distribution; page_distribution.app()
    elif sub == "Tests bivariés":
        from Pages import page_testsbivaries; page_testsbivaries.app()
    elif sub == "Tests multivariés":
        from Pages import page_testsmulti; page_testsmulti.app()

# --- CONTACT ---
elif st.session_state.main_page == "Contact":
    from Pages import page_contact
    page_contact.app()
