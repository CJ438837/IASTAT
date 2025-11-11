import streamlit as st
import pandas as pd
import re
from Bio import Entrez
from io import BytesIO

def app():
    # --- 🌙 Thème Corvus ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Impossible de charger le thème Corvus : {e}")

    st.markdown("""
    **Téléchargez votre fichier afin de visualiser les données, séléctionner les variables utiles**
    **et trouver des études similaires à la vôtre pour vous inspirer**
    """)

    # --- 📂 Page Fichier ---
    st.header("Importez votre jeu de données pour analyse")

    # --- 1️⃣ Upload du fichier ---
    uploaded_file = st.file_uploader(
        "Choisissez votre fichier Excel ou CSV.", 
        type=['xlsx', 'xls', 'csv']
    )

    if uploaded_file:
        # Lecture du fichier
        try:
            if uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.success(f"File '{uploaded_file.name}' successfully loaded!")
        except Exception as e:
            st.error(f"Error while reading the file: {e}")
            return
        
        # --- 2️⃣ Aperçu des données ---
        st.subheader("Data preview")
        st.dataframe(df.head(10), use_container_width=True)

        # --- 3️⃣ Sélection des colonnes ---
        st.subheader("Select columns to include in the study")
        selected_cols = st.multiselect(
            "Check the columns to include",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
        df_selected = df[selected_cols]
        st.write(f"Selected columns ({len(selected_cols)}): {selected_cols}")

        # --- 4️⃣ Description de l'étude ---
        st.subheader("Décrivez votre étude en quelques mots (en anglais)")
        description = st.text_area(
            "Example: Study of the effect of age and weight on blood pressure...",
            placeholder="Enter a short English description..."
        )
        
        # --- 5️⃣ Extraction de mots-clés ---
        if description:
            tokens = re.findall(r'\b\w+\b', description.lower())
            stopwords_en = set([
                "the","a","an","of","and","in","on","for","to","from","by","with","about",
                "as","at","into","that","this","those","these","is","are","be","was","were",
                "it","its","but","or","nor","if","than","so","because","due","such","their",
                "they","he","she","his","her","we","you","your","our","can","may","will",
                "would","should","could","which","has","have","had"
            ])
            keywords = [w for w in tokens if w not in stopwords_en]

            # --- combinaison plus souple pour de meilleurs résultats PubMed ---
            if len(keywords) <= 4:
                query = " AND ".join(keywords)
            elif len(keywords) <= 8:
                query = "(" + " AND ".join(keywords[:3]) + ") AND (" + " OR ".join(keywords[3:]) + ")"
            else:
                query = " OR ".join(keywords)

            st.write(f"**Extracted keywords:** {keywords}")
            st.write(f"**PubMed query:** `{query}`")

            # --- 6️⃣ Recherche PubMed ---
            if st.button("🔍 Search PubMed articles"):
                Entrez.email = "your.email@example.com"  # à remplacer par ton adresse
                try:
                    handle = Entrez.esearch(db="pubmed", term=query, retmax=10, sort="relevance")
                    record = Entrez.read(handle)
                    handle.close()
                    pmids = record['IdList']
                    
                    if not pmids:
                        st.warning("No articles found.")
                    else:
                        st.subheader("Suggested PubMed Articles")
                        for i, pmid in enumerate(pmids, 1):
                            st.markdown(f"{i}. [https://pubmed.ncbi.nlm.nih.gov/{pmid}/](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                except Exception as e:
                    st.error(f"Error during PubMed search: {e}")
        
        # --- 7️⃣ Sauvegarde du DataFrame sélectionné ---
        st.session_state['df_selected'] = df_selected

        # --- 8️⃣ Bouton de navigation ---
        st.markdown("---")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("➡️ Passer à la page Variables", use_container_width=True):
               st.session_state.target_page = "Variables"
               






