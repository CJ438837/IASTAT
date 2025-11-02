import streamlit as st

def app():
    st.header("Fichier")
    st.write("Définissons ton étude")

import streamlit as st
import pandas as pd
from modules.IA_STAT_pubmed import rechercher_pubmed_test

st.title("1️⃣ Import du fichier et contexte de l'étude")

# --- Upload du fichier ---
uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv","xls","xlsx"])

if uploaded_file:
    # Lecture du fichier avec try-except
    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success("✅ Fichier importé avec succès !")
        
        # --- Initialiser st.session_state si pas encore fait ---
        if "data_df" not in st.session_state:
            st.session_state["data_df"] = df

        # --- Sélection des colonnes ---
        st.subheader("Sélection des variables à inclure")
        cols = st.multiselect(
            "Choisissez les colonnes à inclure",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
        if cols:
            st.session_state["data_df"] = df[cols]
            st.dataframe(st.session_state["data_df"])
        else:
            st.warning("⚠️ Vous devez sélectionner au moins une colonne.")

        # --- Contexte de l'étude pour PubMed ---
        st.subheader("Contexte de l'étude")
        description = st.text_area("Décrivez votre étude en quelques phrases :", height=100)
        if st.button("Rechercher articles PubMed"):
            if description.strip():
                liens = rechercher_pubmed_test(description, [])
                if liens:
                    st.markdown("**Articles PubMed suggérés :**")
                    for lien in liens:
                        st.markdown(f"- [{lien}]({lien})")
                else:
                    st.info("Aucun article trouvé pour ce contexte.")
            else:
                st.warning("Veuillez entrer une description de l'étude.")

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
