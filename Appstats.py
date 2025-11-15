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
    st.session_state.theorie_subpage = "Fichier"

# ======================================================
# 🖼️ LOGO
# ======================================================
try:
    logo = Image.open("assets/logo.png")
    st.image(logo, width=1000)
except Exception as e:
    st.warning(f"Logo non trouvé : {e}")

# ======================================================
# 🧭 MENU HORIZONTAL → BANDEAU CORVUS
# ======================================================
MAIN_PAGES = ["Accueil", "Théorie", "Analyse", "Contact"]
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

# --- Styles CSS pour BANDEAU ---
st.markdown("""
<style>
.navbar {
    background-color: #0d1117;
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 25px;
    display: flex;
    justify-content: center;
    gap: 30px;
}

.navbutton {
    padding: 8px 18px;
    border-radius: 6px;
    background-color: #21262d;
    color: #e6edf3;
    font-weight: 500;
    cursor: pointer;
    transition: 0.25s;
    border: 1px solid #30363d;
}

.navbutton:hover {
    background-color: #30363d;
}

.navactive {
    background-color: #238636 !important;
    border-color: #238636 !important;
    color: white !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Construction du bandeau HTML ---
menu_html = '<div class="navbar">'

for page in MAIN_PAGES:
    active_class = "navactive" if st.session_state.main_page == page else ""
    menu_html += f"""
        <form action="" method="post" style="display:inline;">
            <button class="navbutton {active_class}" name="set_page" value="{page}" type="submit">{page}</button>
        </form>
    """

menu_html += "</div>"

# --- Affichage du bandeau ---
st.markdown(menu_html, unsafe_allow_html=True)

# --- Gestion du clic ---
if st.query_params.get("set_page"):
    st.session_state.main_page = st.query_params["set_page"]


# ======================================================
# 🚀 Chargement des pages
# ======================================================
if st.session_state.main_page == "Accueil":
    from Pages import page_accueil
    page_accueil.app()

elif st.session_state.main_page == "Théorie":
    st.subheader("Théorie")

    if st.session_state.theorie_subpage not in THEORIE_PAGES:
        st.session_state.theorie_subpage = THEORIE_PAGES[0]

    st.session_state.theorie_subpage = st.selectbox(
        "Choisir la section:", 
        THEORIE_PAGES,
        index=THEORIE_PAGES.index(st.session_state.theorie_subpage)
    )

    sub = st.session_state.theorie_subpage
    if sub == "Variables":
        from Pages import page_variablesT
        page_variablesT.app()
    elif sub == "Descriptive":
        from Pages import page_descriptiveT
        page_descriptiveT.app()
    elif sub == "Distribution":
        from Pages import page_distributionT
        page_distributionT.app()
    elif sub == "Tests bivariés":
        from Pages import page_bivariesT
        page_bivariesT.app()
    elif sub == "Tests multivariés":
        from Pages import page_multiT
        page_multiT.app()

elif st.session_state.main_page == "Analyse":
    st.subheader("Analyse")

    if st.session_state.analyse_subpage not in ANALYSE_PAGES:
        st.session_state.analyse_subpage = ANALYSE_PAGES[0]

    st.session_state.analyse_subpage = st.selectbox(
        "Choisir l'analyse :", 
        ANALYSE_PAGES,
        index=ANALYSE_PAGES.index(st.session_state.analyse_subpage)
    )

    sub = st.session_state.analyse_subpage
    if sub == "Fichier":
        from Pages import page_fichier
        page_fichier.app()
    elif sub == "Variables":
        from Pages import page_variables
        page_variables.app()
    elif sub == "Descriptive":
        from Pages import page_descriptive
        page_descriptive.app()
    elif sub == "Distribution":
        from Pages import page_distribution
        page_distribution.app()
    elif sub == "Tests bivariés":
        from Pages import page_testsbivaries
        page_testsbivaries.app()
    elif sub == "Tests multivariés":
        from Pages import page_testsmulti
        page_testsmulti.app()

elif st.session_state.main_page == "Contact":
    from Pages import page_contact
    page_contact.app()
