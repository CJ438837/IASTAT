import streamlit as st

# --- 🔧 Thème CORVUS + CSS personnalisé ---
with open("assets/corvus_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Appstats", layout="wide")

# --- ⚙️ Initialisation de la page ---
if "target_page" not in st.session_state:
    st.session_state.target_page = "Accueil"

# --- 🖼️ Logo ---
st.markdown(
    """
    <div style="text-align:center; margin-bottom:10px;">
        <img src="assets/logo.png" style="width:140px;">
    </div>
    """,
    unsafe_allow_html=True
)

# --- 🔘 Barre de navigation horizontale ---
pages = [
    "Accueil", "Fichier", "Variables", "Descriptive",
    "Distribution", "Tests bivariés", "Tests multivariés", "Contact"
]

st.markdown(
    """
    <style>
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 25px;
    }

    .nav-btn {
        background-color: #2b2b2b;
        padding: 8px 18px;
        border-radius: 10px;
        color: white !important;
        border: 1px solid #444;
        font-size: 15px;
        cursor: pointer;
        transition: 0.2s;
    }

    .nav-btn:hover {
        background-color: #3c3c3c;
        border-color: #777;
    }

    .nav-btn-active {
        background-color: #5566ff;
        border-color: #3344cc;
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Construction dynamique des boutons
nav_html = "<div class='nav-container'>"
for page in pages:
    active_class = "nav-btn-active" if st.session_state.target_page == page else ""

    nav_html += (
        f"""
        <button class="nav-btn {active_class}"
            onclick="fetch('?nav={page}', {{method: 'POST'}})
            .then(() => window.location.reload());">
            {page}
        </button>
        """
    )
nav_html += "</div>"

st.markdown(nav_html, unsafe_allow_html=True)

# --- 🔁 Synchronisation navigation ---
nav = st.experimental_get_query_params().get("nav", [None])[0]
if nav and nav in pages and nav != st.session_state.target_page:
    st.session_state.target_page = nav

# --- 🚀 Chargement dynamique des pages ---
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
