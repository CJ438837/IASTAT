import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def descriptive_analysis(df, types_df):
    """
    Réalise une analyse descriptive automatique selon le type de variable.
    
    Args:
        df (DataFrame) : données nettoyées
        types_df (DataFrame) : résultat de detect_variable_types pour cette feuille
    
    Returns:
        dict : clé = variable, valeur = dictionnaire des statistiques descriptives
    """
    summary = {}

    for _, row in types_df.iterrows():
        col = row['variable']
        var_type = row['type']
        col_data = df[col].dropna()  # on ignore les NaN

        if col_data.empty:
            continue

        if var_type == "numérique":
            desc = {
                "n": col_data.count(),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "min": col_data.min(),
                "max": col_data.max(),
                "std": col_data.std(),
                "variance": col_data.var(),
                "quartiles": col_data.quantile([0.25, 0.5, 0.75]).to_dict(),
                "cv": col_data.std() / col_data.mean() if col_data.mean() != 0 else np.nan,
                "skewness": skew(col_data),
                "kurtosis": kurtosis(col_data)
            }

        elif var_type in ["catégorielle", "binaire"]:
            value_counts = col_data.value_counts()
            freqs = (value_counts / col_data.count()).to_dict()
            # Détection des modalités rares (<5% des valeurs)
            rare_categories = [k for k, v in freqs.items() if v < 0.05]

            desc = {
                "n": col_data.count(),
                "modalites": value_counts.to_dict(),
                "frequences_relatives": freqs,
                "modalites_rares": rare_categories
            }

        else:
            desc = {"info": "Type inconnu"}

        summary[col] = desc

    return summary
