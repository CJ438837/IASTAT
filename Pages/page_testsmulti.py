import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.linalg import inv
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA

# -----------------------------------------------------
#  Mardia multivariate normality
# -----------------------------------------------------
def mardia_test(X):
    X = np.asarray(X)
    n, k = X.shape
    X_centered = X - np.mean(X, axis=0)
    S = np.cov(X_centered, rowvar=False)
    invS = inv(S)

    # Skewness
    b1 = 0
    for i in range(n):
        b1 += (X_centered[i] @ invS @ X_centered[i])**3
    b1 = b1 / n / n

    # Kurtosis
    di = np.array([X_centered[i] @ invS @ X_centered[i] for i in range(n)])
    b2 = np.mean(di**2)

    # p-values
    skew_chi = b1 * n / 6
    p_skew = 1 - chi2.cdf(skew_chi, df=int(k*(k+1)*(k+2)/6))

    kurt_z = (b2 - k*(k+2)) / np.sqrt(8*k*(k+2)/n)
    p_kurt = 2 * (1 - chi2.cdf(abs(kurt_z), df=1))

    return {
        "skewness": b1,
        "skew_pvalue": p_skew,
        "kurtosis": b2,
        "kurt_pvalue": p_kurt
    }


# -----------------------------------------------------
#  Box’s M Test
# -----------------------------------------------------
def box_m_test(df, group_col):
    groups = df[group_col].unique()
    k = len(groups)
    n_total = len(df)

    cov_matrices = []
    ns = []

    for g in groups:
        subset = df[df[group_col] == g].drop(columns=[group_col])
        cov_matrices.append(np.cov(subset, rowvar=False))
        ns.append(len(subset))

    pooled_cov = sum([(ns[i] - 1) * cov_matrices[i] for i in range(k)]) / (sum(ns) - k)

    M = (sum(ns) - k) * np.log(np.linalg.det(pooled_cov))
    for i in range(k):
        M -= (ns[i] - 1) * np.log(np.linalg.det(cov_matrices[i]))

    df_box = (df.shape[1] - 1) * (k - 1)
    p_value = 1 - chi2.cdf(M, df=df_box)

    return M, p_value, df_box


# -----------------------------------------------------
#  PAGE STREAMLIT
# -----------------------------------------------------
def app():
    st.title("📊 Tests Multivariés")
    st.markdown("---")

    st.subheader("🎯 Objectif")
    st.markdown("""
    Cette section permet d'analyser **simultanément plusieurs variables quantitatives**  
    et d'étudier leurs relations globales au sein de groupes.
    
    Elle inclut :
    - Normalité multivariée (Mardia)
    - Test d’égalité des matrices de covariance (Box’s M)
    - MANOVA
    - Exploration visuelle
    """)

    st.markdown("---")

    uploaded = st.file_uploader("📥 Importez votre dataset (CSV)", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.markdown("---")

        # -------------------------
        # Multivariate normality
        # -------------------------
        st.subheader("🧪 Normalité multivariée (Test de Mardia)")

        if len(numeric_cols) < 2:
            st.warning("Il faut au moins 2 variables numériques.")
        else:
            try:
                results = mardia_test(df[numeric_cols])

                st.success("Test effectué avec succès")

                st.write(f"**Skewness multivariée** : {results['skewness']:.4f}")
                st.write(f"→ *p-value* = {results['skew_pvalue']:.4f}")

                st.write(f"**Kurtosis multivariée** : {results['kurtosis']:.4f}")
                st.write(f"→ *p-value* = {results['kurt_pvalue']:.4f}")

            except Exception as e:
                st.error(f"Erreur Mardia : {e}")

        st.markdown("---")

        # -------------------------
        # Box's M
        # -------------------------
        st.subheader("🧩 Box’s M")

        group_col = st.selectbox("Sélectionnez la variable de groupe", df.columns)

        if group_col:
            try:
                M, pval, df_box = box_m_test(df[numeric_cols + [group_col]], group_col)

                st.write(f"**Statistique M** : {M:.4f}")
                st.write(f"**df** : {df_box}")
                st.write(f"**p-value** : {pval:.4f}")

                if pval < 0.05:
                    st.warning("⚠️ Les matrices de covariance ne sont pas égales entre les groupes.")
                else:
                    st.success("✔️ Les matrices de covariance sont homogènes entre les groupes.")

            except Exception as e:
                st.error(f"Erreur Box’s M : {e}")

        st.markdown("---")

        # -------------------------
        # MANOVA
        # -------------------------
        st.subheader("📐 MANOVA")

        try:
            formula = " + ".join(numeric_cols) + " ~ " + group_col
            manova = MANOVA.from_formula(formula, data=df)
            st.text(manova.mv_test().summary())
        except Exception as e:
            st.error(f"Erreur MANOVA : {e}")

        st.markdown("---")

        st.markdown("© 2025 Corvus Analytics - Tous droits réservés")

