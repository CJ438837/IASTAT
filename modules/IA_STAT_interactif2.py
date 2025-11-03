import streamlit as st
import pandas as pd
import scipy.stats as stats
import itertools

# ------------------------------------------------------------
# ⚙️ Fonction principale
# ------------------------------------------------------------
def propose_tests_interactif_streamlit(types_df, distribution_df, df, mots_cles):
    """
    Interface Streamlit pour parcourir et exécuter les tests statistiques proposés.
    """

    # --- 1️⃣ Génération de la liste de tests proposés ---
    tests_possibles = []
    numeric_vars = types_df[types_df["type"].str.contains("num", case=False, na=False)]["variable"].tolist()
    cat_vars = types_df[types_df["type"].str.contains("cat", case=False, na=False)]["variable"].tolist()

    # Exemples simples :
    for var1, var2 in itertools.combinations(df.columns, 2):
        if var1 in numeric_vars and var2 in cat_vars:
            tests_possibles.append({
                "id": len(tests_possibles),
                "var1": var1,
                "var2": var2,
                "test": "t-test de Student",
                "description": f"Compare la moyenne de {var1} selon les groupes de {var2}."
            })
        elif var1 in numeric_vars and var2 in numeric_vars:
            tests_possibles.append({
                "id": len(tests_possibles),
                "var1": var1,
                "var2": var2,
                "test": "Corrélation de Pearson",
                "description": f"Mesure la corrélation linéaire entre {var1} et {var2}."
            })
        elif var1 in cat_vars and var2 in cat_vars:
            tests_possibles.append({
                "id": len(tests_possibles),
                "var1": var1,
                "var2": var2,
                "test": "Chi² d’indépendance",
                "description": f"Teste l’indépendance entre {var1} et {var2}."
            })

    if not tests_possibles:
        st.warning("Aucun test statistique pertinent n’a été trouvé.")
        return

    # --- 2️⃣ Initialisation de la session ---
    if "test_index" not in st.session_state:
        st.session_state["test_index"] = 0
    if "results" not in st.session_state:
        st.session_state["results"] = []

    test_courant = tests_possibles[st.session_state["test_index"]]

    # --- 3️⃣ Affichage du test courant ---
    st.markdown("---")
    st.subheader(f"🧪 Test {st.session_state['test_index'] + 1} / {len(tests_possibles)} : {test_courant['test']}")
    st.write(test_courant["description"])
    st.caption(f"Variables : **{test_courant['var1']}** et **{test_courant['var2']}**")

    # --- 4️⃣ Options spécifiques ---
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Seuil de signification (alpha)", 0.01, 0.1, 0.05, step=0.01,
                          key=f"alpha_{test_courant['id']}")
    with col2:
        apparie = st.radio("Apparié ?", ("Non", "Oui"), index=0, key=f"apparie_{test_courant['id']}")

    # --- 5️⃣ Boutons de navigation et d’action ---
    col_prev, col_run, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("⬅️ Précédent", key=f"prev_{test_courant['id']}"):
            if st.session_state["test_index"] > 0:
                st.session_state["test_index"] -= 1
                st.rerun()

    with col_run:
        if st.button("▶️ Exécuter ce test", key=f"run_{test_courant['id']}"):
            resultat = executer_test(df, test_courant, alpha, apparie)
            st.session_state["results"].append(resultat)
            st.success("✅ Test exécuté avec succès !")
            st.write(resultat)
            st.balloons()

    with col_next:
        if st.button("Suivant ➡️", key=f"next_{test_courant['id']}"):
            if st.session_state["test_index"] < len(tests_possibles) - 1:
                st.session_state["test_index"] += 1
                st.rerun()

    # --- 6️⃣ Export des résultats cumulés ---
    if st.session_state["results"]:
        st.markdown("---")
        st.subheader("📈 Résultats cumulés")
        df_res = pd.DataFrame(st.session_state["results"])
        st.dataframe(df_res, use_container_width=True)
        st.download_button(
            "💾 Télécharger les résultats en CSV",
            df_res.to_csv(index=False).encode("utf-8"),
            "resultats_tests.csv",
            "text/csv",
        )

# ------------------------------------------------------------
# 🧠 Fonction d’exécution d’un test
# ------------------------------------------------------------
def executer_test(df, test_courant, alpha, apparie):
    """Exécute le test statistique choisi et renvoie un résumé dict."""
    var1 = test_courant["var1"]
    var2 = test_courant["var2"]
    test = test_courant["test"]

    resultat = {"test": test, "var1": var1, "var2": var2, "alpha": alpha}

    try:
        if test == "t-test de Student":
            groupes = df[var2].dropna().unique()
            if len(groupes) == 2:
                g1 = df[df[var2] == groupes[0]][var1].dropna()
                g2 = df[df[var2] == groupes[1]][var1].dropna()
                stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                resultat.update({"statistique": stat, "p_value": p})
            else:
                resultat.update({"erreur": "Variable catégorielle à plus de 2 groupes."})

        elif test == "Corrélation de Pearson":
            stat, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
            resultat.update({"corrélation": stat, "p_value": p})

        elif test == "Chi² d’indépendance":
            contingency = pd.crosstab(df[var1], df[var2])
            stat, p, _, _ = stats.chi2_contingency(contingency)
            resultat.update({"statistique": stat, "p_value": p})

        # Interprétation simple
        if "p_value" in resultat:
            resultat["significatif"] = "Oui ✅" if resultat["p_value"] < alpha else "Non ❌"

    except Exception as e:
        resultat["erreur"] = str(e)

    return resultat
