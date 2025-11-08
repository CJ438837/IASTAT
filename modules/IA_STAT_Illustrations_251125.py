import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_descriptive(df, types_df, output_folder="plots", selected_vars=None, group_var=None):
    """
    Génère automatiquement les graphiques pour les variables sélectionnées,
    avec option de regroupement.

    Args:
        df (DataFrame): données nettoyées
        types_df (DataFrame): tableau des types détectés
        output_folder (str): dossier de sauvegarde des graphiques
        selected_vars (list): variables à analyser (optionnel)
        group_var (str): variable de regroupement (optionnel)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Si aucune sélection n'est faite → toutes les variables
    if selected_vars is None:
        selected_vars = types_df["variable"].tolist()

    df = df.copy()
    df = df.dropna(subset=selected_vars, how="all")

    # --- 1️⃣ Graphiques univariés ---
    for _, row in types_df.iterrows():
        col = row['variable']
        if col not in selected_vars:
            continue

        var_type = row['type']
        col_data = df[col].dropna()

        if col_data.empty:
            continue

        plt.figure(figsize=(6, 4))
        title = f"{col} ({var_type})"
        if group_var:
            title += f" par {group_var}"

        if var_type == "numérique":
            if group_var and group_var in df.columns:
                sns.boxplot(x=df[group_var], y=df[col], palette="viridis")
            else:
                # histogramme intelligent (entiers vs décimaux)
                if np.all(col_data.dropna() == col_data.dropna().astype(int)):
                    bins = np.arange(col_data.min() - 0.5, col_data.max() + 1.5, 1)
                    sns.histplot(col_data, bins=bins, color='skyblue', kde=False)
                else:
                    sns.histplot(col_data, kde=True, bins=20, color='skyblue')

            plt.xlabel(col)
            plt.ylabel("Fréquence")

        elif var_type in ["catégorielle", "binaire"]:
            if group_var and group_var in df.columns:
                sns.countplot(x=df[col], hue=df[group_var], palette="Set2")
            else:
                sns.countplot(x=col_data, palette="Set2")
            plt.xlabel(col)
            plt.ylabel("Effectif")

        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{col}_univariate.png")
        plt.close()

    # --- 2️⃣ Graphiques bivariés ---
    num_cols = [v for v in selected_vars if v in types_df[types_df['type'] == "numérique"]['variable'].tolist()]
    cat_cols = [v for v in selected_vars if v in types_df[types_df['type'].isin(["catégorielle", "binaire"])]['variable'].tolist()]

    # Numérique vs numérique → scatterplots
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            x, y = num_cols[i], num_cols[j]
            if x not in df.columns or y not in df.columns:
                continue

            plt.figure(figsize=(6, 4))
            if group_var and group_var in df.columns:
                sns.scatterplot(x=df[x], y=df[y], hue=df[group_var], palette="viridis")
            else:
                sns.scatterplot(x=df[x], y=df[y], color="#3B82F6")
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y}" + (f" par {group_var}" if group_var else ""))
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{x}_vs_{y}_scatter.png")
            plt.close()

    # Numérique vs catégorielle → boxplots
    for num in num_cols:
        for cat in cat_cols:
            if num not in df.columns or cat not in df.columns:
                continue

            plt.figure(figsize=(6, 4))
            if group_var and group_var in df.columns and group_var != cat:
                sns.boxplot(x=df[cat], y=df[num], hue=df[group_var], palette="Set3")
            else:
                sns.boxplot(x=df[cat], y=df[num], palette="Set3")
            plt.xlabel(cat)
            plt.ylabel(num)
            plt.title(f"{num} vs {cat}" + (f" par {group_var}" if group_var else ""))
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{num}_vs_{cat}_boxplot.png")
            plt.close()

    # --- 3️⃣ Matrice de corrélation ---
    if len(num_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matrice de corrélation (variables sélectionnées)")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/correlation_heatmap.png")
        plt.close()
