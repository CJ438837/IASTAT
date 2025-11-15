import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

plt.style.use("seaborn-v0_8-muted")


def app():

    # --- 🎨 Thème Corvus ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Impossible de charger le thème Corvus : {e}")

    # --- 🧠 En-tête ---
    st.markdown("<h1 class='corvus-title'> Tests Multivariés Avancés</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Passons maintenant aux interactions plus complexes !**  
    **Ici l'étude se fait avec une variable dépendante et plusieurs variables explicatives.**
    """)

    # --- 1️⃣ Vérification des prérequis ---
    if "df_selected" not in st.session_state or st.session_state["df_selected"] is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier dans l'onglet **Fichier**.")
        st.stop()
    df = st.session_state["df_selected"]

    if "types_df" not in st.session_state or st.session_state["types_df"] is None:
        types_df = pd.DataFrame({
            "variable": df.columns,
            "type": [
                "numérique" if pd.api.types.is_numeric_dtype(df[col]) else "catégorielle"
                for col in df.columns
            ]
        })
        st.session_state["types_df"] = types_df
    else:
        types_df = st.session_state["types_df"]

    st.success(f"✅ Données disponibles : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # --- 2️⃣ Aperçu ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Aperçu du jeu de données")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 3️⃣ Sélection ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Sélection des variables")

    target_var = st.selectbox("Variable à expliquer :", df.columns)
    explicatives = st.multiselect(
        "Variables explicatives :",
        [c for c in df.columns if c != target_var],
        default=[]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not explicatives:
        st.info("💡 Sélectionnez au moins une variable explicative pour continuer.")
        st.stop()

    # --- 4️⃣ Lancer les tests ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Lancez l'analyse multivariée")

    if st.button("📈 Démarrer l'analyse multivariée", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            try:
                results = propose_tests_multivariés(
                    df,
                    types_df,
                    target_var=target_var,
                    explicatives=explicatives
                )

                # --- Boucle sur les résultats ---
                for res in results:
                    st.divider()
                    st.subheader(f"🧩 {res.get('test', 'Test inconnu')}")

                    # ⚠️ Gestion d'erreur
                    if "error" in res:
                        st.error(f"❌ Erreur : {res['error']}")
                        continue
                    if "message" in res:
                        st.warning(res["message"])
                        continue

                    # 📄 Résultats tabulaires
                    result_df = res.get("result_df")
                    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                        st.markdown("#### 📊 Tableau de résultats")
                        st.dataframe(result_df, use_container_width=True)

                    # 📊 Graphique
                    fig = res.get("fig")
                    if fig is not None:
                        st.markdown("#### 📈 Graphique associé")
                        st.pyplot(fig)

                    # 🔍 Informations avancées
                    if "info" in res and res["info"]:
                        st.markdown("#### 🧠 Informations additionnelles")
                        info = res["info"]

                        # JSON propre + conversion numpy -> python
                        clean_info = {}

                        for k, v in info.items():
                            try:
                                if isinstance(v, pd.DataFrame):
                                    clean_info[k] = v.to_dict(orient="records")
                                elif hasattr(v, "tolist"):
                                    clean_info[k] = v.tolist()
                                else:
                                    clean_info[k] = v
                            except Exception:
                                clean_info[k] = str(v)

                        st.json(clean_info)

                    # 💬 Interprétation auto
                    if "interpretation" in res and res["interpretation"]:
                        st.markdown(
                            f"<div class='corvus-interpretation'><b>Interprétation :</b> {res['interpretation']}</div>",
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {e}")

    st.markdown("</div>", unsafe_allow_html=True)
