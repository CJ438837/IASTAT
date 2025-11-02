import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_descriptive(df, types_df, output_folder="plots"):
    """
    Génère automatiquement les graphiques pour les variables détectées.
    
    Args:
        df (DataFrame) : données nettoyées
        types_df (DataFrame) : tableau des types détectés
        output_folder (str) : dossier où enregistrer les graphiques
    """
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- 1️⃣ Graphiques univariés ---
    for _, row in types_df.iterrows():
        col = row['variable']
        var_type = row['type']
        col_data = df[col].dropna()

        if col_data.empty:
            continue

        plt.figure(figsize=(6,4))
        plt.title(f"{col} ({var_type})")

        if var_type == "numérique":
    # Vérifie si toutes les valeurs sont des entiers
            if np.all(col_data.dropna() == col_data.dropna().astype(int)):
                bins = np.arange(col_data.min() - 0.5, col_data.max() + 1.5, 1)  # un bin par entier
                sns.histplot(col_data, bins=bins, color='skyblue', kde=False)
            else:
                sns.histplot(col_data, kde=True, bins=20, color='skyblue')
            plt.xlabel(col)
            plt.ylabel("Fréquence")

        elif var_type in ["catégorielle", "binaire"]:
            sns.countplot(x=col_data, palette="Set2")
            plt.xlabel(col)
            plt.ylabel("Effectif")
        
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{col}_univariate.png")
        plt.close()

    # --- 2️⃣ Graphiques bivariés ---
    num_cols = types_df[types_df['type']=="numérique"]['variable'].tolist()
    cat_cols = types_df[types_df['type'].isin(["catégorielle","binaire"])]['variable'].tolist()

    # Numérique vs numérique → scatterplots
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            x, y = num_cols[i], num_cols[j]
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=df[x], y=df[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y}")
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{x}_vs_{y}_scatter.png")
            plt.close()

    # Numérique vs catégorielle → boxplots
    for num in num_cols:
        for cat in cat_cols:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[cat], y=df[num], palette="Set3")
            plt.xlabel(cat)
            plt.ylabel(num)
            plt.title(f"{num} vs {cat}")
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{num}_vs_{cat}_boxplot.png")
            plt.close()

    # Matrice de corrélation pour variables numériques
    if len(num_cols) > 1:
        plt.figure(figsize=(8,6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/correlation_heatmap.png")
        plt.close()
