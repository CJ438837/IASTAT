import pandas as pd
import numpy as np

def detect_variable_types(file_path: str, sheet_name=None):
    """
    Détecte automatiquement le type des variables dans un fichier CSV ou Excel.
    Types possibles : numérique, catégorielle, binaire.
    Gère les fichiers Excel à feuille unique ou multiple.
    Les valeurs manquantes sont ignorées dans l’analyse.
    
    Args:
        file_path (str): chemin vers le fichier CSV ou Excel
        sheet_name (str/int, optional): nom ou index de la feuille (Excel). 
                                        Si None, toutes les feuilles sont traitées.
    
    Returns:
        dict: clé = nom de la feuille (ou 'data'), valeur = DataFrame des types détectés
        dict: clé = nom de la feuille, valeur = DataFrame correspondant
    """
    
    # --- 1️⃣ Lecture du fichier ---
    if file_path.endswith(('.xls', '.xlsx')):
        if sheet_name is None:
            all_sheets = pd.read_excel(file_path, sheet_name=None)
        else:
            all_sheets = {sheet_name: pd.read_excel(file_path, sheet_name=sheet_name)}
    elif file_path.endswith('.csv'):
        all_sheets = {'data': pd.read_csv(file_path, sep=None, engine='python')}
    else:
        raise ValueError("Format non supporté. Utiliser .csv, .xls ou .xlsx")

    types_results = {}
    cleaned_data = {}

    # --- 2️⃣ Boucle sur chaque feuille ---
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
