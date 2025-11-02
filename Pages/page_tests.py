import streamlit as st

def app():
    st.header("Tests")
    st.write("Le dur du sujet ! voyons ce que tes données ont dans le ventre")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from Bio import Entrez
import itertools
import numpy as np
from modules.IA_STAT_interactif2 import propose_tests_interactif

def app():
    st.title("📊 Tests statistiques interactifs")

    # --- Vérifications préalables ---
    if "df_selected" not in st.session_state or "types_df" not in st.session_state or "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier et détecter les types de variables dans les pages Fichier et Variables.")
        return

    df = st.session_state["df_selected"]
    types_df = st.session_state["types_df"]
    distribution_df = st.session_state["distribution_df"]
    mots_cles = st.session_state.get("keywords", [])

    st.markdown("### 💡 Propositions de tests")
    
    # --- Checkbox pour lancer les tests ---
    lancer_tests = st.button("Lancer les tests interactifs")
    
    if lancer_tests:
        # Appel à la fonction interactive adaptée
        propose_tests_interactif(types_df, distribution_df, df, mots_cles)
        st.success("✅ Tous les tests interactifs ont été proposés et exécutés.")
