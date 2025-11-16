# Pages/page_testsmulti.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

plt.style.use("seaborn-v0_8-muted")

def _display_info(info):
    """Affiche proprement le dict 'info' renvoyé par le module."""
    if info is None:
        return
    if not isinstance(info, dict):
        st.write(info)
        return
    # DataFrame present as 'table' key
    if "table" in info and isinstance(info["table"], list):
        try:
            df = pd.DataFrame(info["table"])
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.write(info["table"])
        return
    # Otherwise print keys
    for k, v in info.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            st.markdown(f"**{k}** :")
            try:
                st.dataframe(pd.DataFrame(v), use_container_width=True)
            except Exception:
                st.write(v)
        else:
            st.write(f"- **{k}** : {v}")


def app():
    st.title("📊 Tests Multivariés — Interface")
    st.markdown("""
    Ici vous pouvez lancer les analyses multivariées : PCA, MCA, FAMD, MANOVA, régression multiple, diagnostics, corrélations, normalité multivariée et Box's M (lorsque possible).
    """)

    # prérequis
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types dans la page Variables.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()

    st.success(f"✅ Données prêtes — {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # sélection variables
    st.header("🎯 Sélection des variables")
    cols = df.columns.tolist()
    col1, col2 = st.columns([2, 3])
    with col1:
        target_var = st.selectbox("Variable cible (à expliquer)", cols)
    with col2:
        explicatives = st.multiselect("Variables explicatives (sélectionnez 1+)", [c for c in cols if c != target_var])

    if not explicatives:
        st.info("Sélectionnez au moins une variable explicative.")
        st.stop()

    # dossier figures (optionnel)
    output_folder = st.text_input("Dossier pour enregistrer les figures (optionnel)", value="multivaries_plots")

    if st.button("📈 Lancer l'analyse multivariée"):
        with st.spinner("Exécution..."):
            try:
                results = propose_tests_multivariés(df, types_df, target_var, explicatives)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'exécution : {e}")
                st.stop()

        st.success("✅ Analyse terminée")

        # affichage résultats
        for res in results:
            test_name = res.get("test", "Test inconnu")
            st.subheader(f"🧪 {test_name}")

            if res.get("error"):
                st.error(f"Erreur : {res['error']}")
                st.markdown("---")
                continue

            # tableau principal
            df_res = res.get("result_df")
            if isinstance(df_res, pd.DataFrame) and not df_res.empty:
                st.markdown("**Résultats**")
                st.dataframe(df_res, use_container_width=True)

            # figure
            fig = res.get("fig")
            if fig is not None:
                try:
                    st.markdown("**Graphique**")
                    st.pyplot(fig)
                except Exception:
                    st.info("Figure présente mais impossible à afficher.")

            # info (toujours dict via module)
            info = res.get("info", {})
            if info:
                st.markdown("**Informations complémentaires**")
                _display_info(info)

            # interpretation
            interp = res.get("interpretation")
            if interp:
                st.markdown("**Interprétation**")
                st.info(interp)

            # save figure (best effort)
            if fig is not None and output_folder:
                try:
                    os.makedirs(output_folder, exist_ok=True)
                    safe_name = f"{test_name.replace(' ', '_').replace('/', '_')}.png"
                    path = os.path.join(output_folder, safe_name)
                    try:
                        fig.savefig(path, bbox_inches="tight")
                    except Exception:
                        pass
                except Exception:
                    pass

            st.markdown("---")

    # navigation retour (comme ta page bivariée)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⬅️ Retour : Tests bivariés"):
            st.session_state.main_page = "Analyse"
            st.session_state.analyse_subpage = "Tests bivariés"
