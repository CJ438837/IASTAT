import streamlit as st

def app():
    st.header("Introduction")
    st.write("Définissons ton projet")

import streamlit as st
import pandas as pd
from modules.IA_STAT_pubmed import rechercher_pubmed_test  # ta fonction PubMed

st.title("1️⃣ Import du fichier et contexte de l'étude")

# --- Upload du fichier ---
uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv","xls","xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success("✅ Fichier importé avec succès !")
        st.session_state["data_df"] = df

        # --- Sélection des colonnes à inclure ---
        st.subheader("Sélection des variables à inclure dans l'étude")
        cols = st.multiselect(
            "Sélectionnez les colonnes à inclure",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
        st.session_state["data_df"] = df[cols]
        st.dataframe(st.session_state["data_df"])

        # --- Contexte de l'étude pour PubMed ---
        st.subheader("Contexte de l'étude")
        description = st.text_area(
            "Décrivez votre étude en quelques phrases :",
            height=100
        )
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


