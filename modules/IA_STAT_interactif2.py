import streamlit as st
import pandas as pd
import scipy.stats as stats
import itertools


def propose_tests_interactif_streamlit(types_df, distribution_df, df, mots_cles=None, generation_mode=False):

    """
    Interface Streamlit pour parcourir les tests proposés sans rechargement de page.
    """

    # --- 1️⃣ Construction des tests possibles ---
    numeric_vars = types_df[types_df["type"].str.contains("num", case=False, na=False)]["variable"].tolist()
    cat_vars = types_df[types_df["type"].str.contains("cat", case=False, na=False)]["variable"].tolist()

    tests_possibles = []
    for var1, var2 in itertools.combinations(df.columns, 2):
        if var1 in numeric_vars and var2 in cat_vars:
            tests_possibles.append({
                "var1": var1, "var2": var2, "test": "t-test de Student",
                "description": f"Compare la moyenne de {var1} selon les groupes de {var2}."
            })
        elif var1 in numeric_vars and var2 in numeric_vars:
            tests_possibles.append({
                "var1": var1, "var2": var2, "test": "Corrélation de Pearson",
                "description": f"Mesure la corrélation entre {var1} et {var2}."
            })
        elif var1 in cat_vars and var2 in cat_vars:
            tests_possibles.append({
                "var1": var1, "var2": var2, "test": "Chi² d’indépendance",
                "description": f"Teste l’indépendance entre {var1} et {var2}."
            })

    if not tests_possibles:
        st.warning("Aucun test statistique pertinent n’a été trouvé.")
        return

    # --- 2️⃣ État persisté (mais sans rechargement forcé) ---
    if "test_index" not in st.session_state:
        st.session_state.test_index = 0
    if "results" not in st.session_state:
        st.session_state.results = []

    # Sélection du test courant
    i = st.session_state.test_index
    test_courant = tests_possibles[i]

    # --- 3️⃣ Affichage du test courant ---
    st.markdown("---")
    st.subheader(f"🧪 Test {i + 1} / {len(tests_possibles)} : {test_courant['test']}")
    st.write(test_courant["description"])
    st.caption(f"Variables : **{test_courant['var1']}** et **{test_courant['var2']}**")

    # --- 4️⃣ Choix des paramètres ---
    alpha = st.slider(
        "Seuil de signification (alpha)", 0.01, 0.1, 0.05, step=0.01,
        key=f"alpha_{i}"
    )
    apparie = st.radio(
        "Apparié ?", ("Non", "Oui"), index=0,
        key=f"apparie_{i}"
    )

    # --- 5️⃣ Boutons sans rechargement ---
    col_prev, col_run, col_next = st.columns([1, 2, 1])

    if col_prev.button("⬅️ Précédent", key=f"prev_{i}", use_container_width=True):
        if i > 0:
            st.session_state.test_index -= 1

    if col_next.button("Suivant ➡️", key=f"next_{i}", use_container_width=True):
        if i < len(tests_possibles) - 1:
            st.session_state.test_index += 1

    # Exécution du test sans recharger
    if col_run.button("▶️ Exécuter ce test", key=f"run_{i}", use_container_width=True):
        resultat = executer_test(df, test_courant, alpha, apparie)
        st.session_state.results.append(resultat)
        st.session_state.last_result = resultat

    # --- 6️⃣ Affichage du dernier résultat ---
    if "last_result" in st.session_state:
        st.markdown("### 🧾 Résultat du test")
        res = st.session_state.last_result
        st.write(pd.DataFrame([res]))

    # --- 7️⃣ Tableau cumulatif des résultats ---
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📊 Résultats cumulés")
        df_res = pd.DataFrame(st.session_state.results)
        st.dataframe(df_res, use_container_width=True)
        st.download_button(
            "💾 Télécharger les résultats",
            df_res.to_csv(index=False).encode("utf-8"),
            "resultats_tests.csv",
            "text/csv"
        )


def executer_test(df, test_courant, alpha, apparie):
    """Exécute un test statistique et retourne un dictionnaire de résultat."""
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
                resultat["erreur"] = "Variable catégorielle à plus de 2 groupes."

        elif test == "Corrélation de Pearson":
            stat, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
            resultat.update({"corrélation": stat, "p_value": p})

        elif test == "Chi² d’indépendance":
            contingency = pd.crosstab(df[var1], df[var2])
            stat, p, _, _ = stats.chi2_contingency(contingency)
            resultat.update({"statistique": stat, "p_value": p})

        if "p_value" in resultat:
            resultat["significatif"] = "Oui ✅" if resultat["p_value"] < alpha else "Non ❌"

    except Exception as e:
        resultat["erreur"] = str(e)

    return resultat
