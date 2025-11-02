import pandas as pd
import numpy as np
from typing import Union

def detect_variable_types(input_data: Union[str, pd.DataFrame], sheet_name=None):
    """
    Détecte automatiquement le type des variables dans un fichier CSV/Excel
    ou dans un DataFrame déjà chargé.
    Types possibles : numérique, catégorielle, binaire.
    Les valeurs manquantes sont ignorées dans l’analyse.
    
    Args:
        input_data (str or pd.DataFrame): chemin vers le fichier CSV/Excel ou DataFrame
        sheet_name (str/int, optional): nom ou index de la feuille (Excel). 
                                        Si None, toutes les feuilles sont traitées.
    
    Returns:
        dict: clé = nom de la feuille (ou 'data'), valeur = DataFrame des types détectés
        dict: clé = nom de la feuille, valeur = DataFrame correspondant
    """

    # --- 1️⃣ Lecture du fichier si input_data est un chemin ---
    if isinstance(input_data, pd.DataFrame):
        all_sheets = {'data': input_data.copy()}
    elif isinstance(input_data, str):
        if input_data.endswith(('.xls', '.xlsx')):
            if sheet_name is None:
                all_sheets = pd.read_excel(input_data, sheet_name=None)
            else:
                all_sheets = {sheet_name: pd.read_excel(input_data, sheet_name=sheet_name)}
        elif input_data.endswith('.csv'):
            all_sheets = {'data': pd.read_csv(input_data, sep=None, engine='python')}
        else:
            raise ValueError("Format non supporté. Utiliser .csv, .xls ou .xlsx")
    else:
        raise TypeError("input_data doit être un chemin de fichier ou un DataFrame")

    types_results = {}
    cleaned_data = {}

    # --- 2️⃣ Boucle sur chaque feuille / DataFrame ---
    for sheet, df in all_sheets.items():
        df = df.dropna(axis=1, how='all')  # supprime colonnes complètement vides
        results = []

        for col in df.columns:
            col_data = df[col].dropna()
            if col_data.empty:
                continue

            # Normalisation pour l'analyse des valeurs uniques
            unique_vals = pd.Series(col_data).astype(str).str.strip().unique()
            n_unique = len(unique_vals)

            # --- Détection du type ---
            if n_unique == 2:
                var_type = "binaire"
            elif np.issubdtype(col_data.dtype, np.number):
                var_type = "numérique"
            else:
                var_type = "catégorielle"

            results.append({
                "variable": col,
                "type": var_type,
                "valeurs_uniques": n_unique,
                "exemples": unique_vals[:5]
            })

        types_results[sheet] = pd.DataFrame(results)
        cleaned_data[sheet] = df

    return types_results, cleaned_data
