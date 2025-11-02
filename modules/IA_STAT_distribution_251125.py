import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, probplot
from fitter import Fitter

def advanced_distribution_analysis(df, types_df, output_folder="distribution_plots"):
    """
    Analyse la distribution des variables numériques et propose les distributions les plus probables.

    Args:
        df (DataFrame) : données nettoyées
        types_df (DataFrame) : tableau des types détectés
        output_folder (str) : dossier où enregistrer les graphiques

    Returns:
        DataFrame : tableau récapitulatif des tests de normalité et distribution la plus probable
    """
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = []
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()

    for col in num_vars:
        col_data = df[col].dropna()
        n = len(col_data)
        if n == 0:
            continue

        # --- 1️⃣ Test de normalité ---
        if n < 5000:
            stat, p_value = shapiro(col_data)
            test_used = "Shapiro-Wilk"
        else:
            stat, p_value = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
            test_used = "Kolmogorov-Smirnov"

        verdict = "Normal" if p_value > 0.05 else "Non Normal"

        # --- 2️⃣ Détection distribution probable ---
        try:
            if np.all(col_data == col_data.astype(int)):
                # Discrètes → poisson, binomiale
                f = Fitter(col_data, distributions=['poisson', 'binom'])
            else:
                # Continues → normale, uniforme, exponentielle, lognormale
                f = Fitter(col_data, distributions=['norm', 'expon', 'lognorm', 'uniform'])
            f.fit()

            summary_df = f.summary()
            if not summary_df.empty:
                if 'Distribution' in summary_df.columns:
                    best_fit = summary_df['Distribution'].iloc[0]
                elif 'distribution' in summary_df.columns:
                    best_fit = summary_df['distribution'].iloc[0]
                else:
                    best_fit = "unknown"
            else:
                best_fit = "unknown"
        except Exception as e:
            best_fit = "unknown"

        results.append({
            "variable": col,
            "n": n,
            "normality_test": test_used,
            "statistic": stat,
            "p_value": p_value,
            "verdict": verdict,
            "best_fit_distribution": best_fit
        })

        # --- 3️⃣ Graphiques ---
        plt.figure(figsize=(12,5))
        # Histogramme + KDE
        plt.subplot(1,2,1)
        if np.all(col_data == col_data.astype(int)):
            bins = np.arange(col_data.min()-0.5, col_data.max()+1.5, 1)
            sns.histplot(col_data, bins=bins, kde=False, color='skyblue')
        else:
            sns.histplot(col_data, bins=20, kde=True, color='skyblue')
        plt.title(f"{col} - Histogramme + KDE")
        plt.xlabel(col)
        plt.ylabel("Fréquence")

        # QQ plot
        plt.subplot(1,2,2)
        probplot(col_data, dist="norm", plot=plt)
        plt.title(f"{col} - QQ Plot")

        plt.tight_layout()
        plt.savefig(f"{output_folder}/{col}_distribution.png")
        plt.close()

    return pd.DataFrame(results)
