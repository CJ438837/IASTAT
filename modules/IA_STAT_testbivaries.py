import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import numpy as np

def propose_tests_bivariés(df, types_df, distribution_df):
    """Propose tous les tests bivariés automatiquement, un par un."""
    
    num_vars = types_df[types_df['type']=="numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()
    
    test_list = []
    
    # 1️⃣ Numérique vs Catégoriel
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]
        
        if n_modalites == 2:
            test_name = "t-test" if verdict=="Normal" else "Mann-Whitney"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict=="Normal" else "Kruskal-Wallis"
        else:
            test_name = "unknown"
        
        groupes = df.groupby(cat)[num].apply(list)
        apparie_needed = test_name in ["t-test","Mann-Whitney"]
        
        # DataFrame résultat par test
        result_df = pd.DataFrame([{
            "Variable_num": num,
            "Variable_cat": cat,
            "Test": test_name,
            "Apparié": None if apparie_needed else False,
            "Statistique": None if apparie_needed else 0,
            "p-value": None if apparie_needed else 0
        }])
        
        test_list.append({
            "test_name": test_name,
            "num": num,
            "cat": cat,
            "groupes": groupes,
            "apparie_needed": apparie_needed,
            "result_df": result_df
        })
    
    # 2️⃣ Deux variables numériques
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"
        
        if test_type=="Pearson":
            corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
        else:
            corr, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
        
        result_df = pd.DataFrame([{
            "Variable_num1": var1,
            "Variable_num2": var2,
            "Test": f"Corrélation ({test_type})",
            "Statistique": corr,
            "p-value": p
        }])
        
        test_list.append({
            "test_name": f"Corrélation ({test_type})",
            "var1": var1,
            "var2": var2,
            "result_df": result_df,
            "apparie_needed": False
        })
    
    # 3️⃣ Deux variables catégorielles
    for var1, var2 in itertools.combinations(cat_vars, 2):
        contingency_table = pd.crosstab(df[var1], df[var2])
        try:
            if contingency_table.size <= 4:
                stat, p = stats.fisher_exact(contingency_table)
                test_name = "Fisher exact"
            else:
                stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                test_name = "Chi²"
        except Exception:
            stat, p = None, None
            test_name = "Chi² / Fisher"
        
        result_df = pd.DataFrame([{
            "Variable_cat1": var1,
            "Variable_cat2": var2,
            "Test": test_name,
            "Statistique": stat,
            "p-value": p
        }])
        
        test_list.append({
            "test_name": test_name,
            "var1": var1,
            "var2": var2,
            "contingency_table": contingency_table,
            "result_df": result_df,
            "apparie_needed": False
        })
    
    return test_list


def app():
    st.title("📊 Tests statistiques bivariés")
    
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser la distribution des données dans la page Distribution.")
        st.stop()
    
    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    
    lancer_tests = st.button("🧠 Exécuter tous les tests bivariés")
    
    if lancer_tests:
        with st.spinner("Exécution des tests bivariés... ⏳"):
            test_list = propose_tests_bivariés(df, types_df, distribution_df)
            
            for i, test_data in enumerate(test_list):
                st.markdown(f"### 🔹 Test {i+1} : {test_data['test_name']}")
                
                # 1️⃣ Tests appariés
                if test_data.get("apparie_needed", False):
                    apparie_choice = st.radio(f"Le test {test_data['test_name']} est-il apparié ?", ("Non","Oui"), key=f"apparie_{i}")
                    apparie = apparie_choice=="Oui"
                    
                    g = test_data["groupes"]
                    try:
                        if test_data["test_name"]=="t-test":
                            stat, p = (stats.ttest_rel(g.iloc[0], g.iloc[1]) if apparie else stats.ttest_ind(g.iloc[0], g.iloc[1]))
                        elif test_data["test_name"]=="Mann-Whitney":
                            stat, p = (stats.wilcoxon(g.iloc[0], g.iloc[1]) if apparie else stats.mannwhitneyu(g.iloc[0], g.iloc[1]))
                        else:
                            stat, p = None, None
                    except Exception as e:
                        st.error(f"❌ Erreur pendant l'exécution de {test_data['test_name']} : {e}")
                        stat, p = None, None
                    
                    test_data["result_df"].at[0, "Apparié"] = apparie
                    test_data["result_df"].at[0, "Statistique"] = stat
                    test_data["result_df"].at[0, "p-value"] = p
                    
                    # Boxplot
                    fig, ax = plt.subplots()
                    sns.boxplot(x=test_data['cat'], y=test_data['num'], data=df, ax=ax)
                    ax.set_title(f"{test_data['num']} vs {test_data['cat']} ({test_data['test_name']})")
                    st.pyplot(fig)
                    plt.close(fig)
                
                # 2️⃣ Tests non appariés (corrélation et chi²)
                else:
                    if test_data['test_name'].startswith("Corrélation"):
                        fig, ax = plt.subplots()
                        ax.scatter(df[test_data['var1']], df[test_data['var2']], alpha=0.6)
                        ax.set_xlabel(test_data['var1'])
                        ax.set_ylabel(test_data['var2'])
                        ax.set_title(f"{test_data['var1']} vs {test_data['var2']} ({test_data['test_name']})")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        # Chi² ou Fisher
                        fig, ax = plt.subplots()
                        sns.heatmap(test_data['contingency_table'], annot=True, fmt="d", cmap="coolwarm", ax=ax)
                        ax.set_title(f"{test_data.get('var1','')} vs {test_data.get('var2','')} ({test_data['test_name']})")
                        st.pyplot(fig)
                        plt.close(fig)
                
                # Affichage tableau résultat
                st.dataframe(test_data['result_df'])
