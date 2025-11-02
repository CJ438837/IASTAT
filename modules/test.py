import re
from googletrans import Translator
from Bio import Entrez

# --- Config PubMed ---
Entrez.email = "ton.email@example.com"

# --- Entrée utilisateur ---
description = input("Décrivez votre étude en quelques phrases : ")

# --- Extraction de mots alphabétiques (évite NLTK) ---
tokens = re.findall(r'\b\w+\b', description.lower())
stopwords_fr = set([
    "le","la","les","un","une","des","de","du","et","en","au","aux","avec",
    "pour","sur","dans","par","au","a","ce","ces","est","sont","ou","où","se",
    "sa","son","que","qui","ne","pas","plus","moins","comme","donc"
])
keywords_fr = [w for w in tokens if w not in stopwords_fr]

# --- Traduction en anglais ---
translator = Translator()
keywords_en = [translator.translate(word, src='fr', dest='en').text for word in keywords_fr]


print(f"\nMots-clés français : {keywords_fr}")
print(f"Mots-clés traduits anglais : {keywords_en}")





from IA_STAT_typevariable_251125 import detect_variable_types

# Remplace par le chemin vers ton fichier
file_path = "C:/Users/cedri/Downloads/Poids_Decembre_AV.xlsx"

# Détecte les types
types_df, df = detect_variable_types(file_path)

# Affiche le résultat
print("\n=== Types détectés ===")
print(types_df)


from IA_STAT_descriptive_251125 import descriptive_analysis  # la fonction que tu as codée

# --- 1️⃣ Fichier Excel à tester ---
file_path = "C:/Users/cedri/Downloads/Poids_Decembre_AV.xlsx"

# --- 2️⃣ Détecte les types de variables ---
types_dict, data_dict = detect_variable_types(file_path)

# Pour cet exemple, on prend la première feuille
sheet_name = list(types_dict.keys())[0]  # si Excel, sinon 'data' pour CSV
types_df = types_dict[sheet_name]
df_sheet = data_dict[sheet_name]

# --- 3️⃣ Analyse descriptive ---
summary = descriptive_analysis(df_sheet, types_df)

# --- 4️⃣ Affichage lisible des résultats ---
for var, stats in summary.items():
    print(f"\n--- Variable : {var} ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


# test_visualisation.py

from IA_STAT_Illustrations_251125 import plot_descriptive  # ta fonction d'illustrations

# --- 1️⃣ Fichier Excel à tester ---
file_path = "C:/Users/cedri/Downloads/Poids_Decembre_AV.xlsx"

# --- 2️⃣ Détecte les types de variables ---
types_dict, data_dict = detect_variable_types(file_path)

# Pour cet exemple, on prend la première feuille
sheet_name = list(types_dict.keys())[0]  # si Excel, sinon 'data' pour CSV
types_df = types_dict[sheet_name]
df_sheet = data_dict[sheet_name]

# --- 3️⃣ Génération des graphiques ---
output_folder = "D:/Programation/IA stat/plots_test"  # dossier où les graphiques seront sauvegardés
plot_descriptive(df_sheet, types_df, output_folder=output_folder)

print(f"\n=== Graphiques générés dans le dossier : {output_folder} ===")


# test_distribution.py

from IA_STAT_distribution_251125 import advanced_distribution_analysis  # ta fonction avancée

# --- 1️⃣ Fichier Excel à tester ---
file_path = "C:/Users/cedri/Downloads/Poids_Decembre_AV.xlsx"

# --- 2️⃣ Détecte les types de variables ---
types_dict, data_dict = detect_variable_types(file_path)

# Pour cet exemple, on prend la première feuille
sheet_name = list(types_dict.keys())[0]  # si Excel, sinon 'data' pour CSV
types_df = types_dict[sheet_name]
df_sheet = data_dict[sheet_name]

# --- 3️⃣ Analyse de distribution avancée ---
output_folder = "D:/Programation/IA stat/distribution_test"  # dossier où les graphiques seront sauvegardés
distribution_df = advanced_distribution_analysis(df_sheet, types_df, output_folder=output_folder)

# --- 4️⃣ Affichage des résultats ---
print("\n=== Résultat analyse distribution avancée ===")
print(distribution_df)

print(f"\nGraphiques générés dans le dossier : {output_folder}")


# test_tests_statistiques.py

from IA_STAT_interactif2 import propose_tests_interactif  # nouvelle fonction interactive

# Pour cet exemple, on prend la première feuille
sheet_name = list(types_dict.keys())[0]  # si Excel, sinon 'data' pour CSV
types_df = types_dict[sheet_name]
df_sheet = data_dict[sheet_name]

# Définir une liste de mots-clés pour la recherche PubMed
mots_cles = keywords_en

# --- 4️⃣ Proposition et exécution interactive des tests ---
propose_tests_interactif(types_df, distribution_df, df_sheet, mots_cles)


