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

    # --- 📂 Page Fichier ---
    st.header("📁 Importer le fichier pour l'étude")
    
    # --- 1️⃣ Upload du fichier ---
    uploaded_file = st.file_uploader(
        "Choisissez votre fichier Excel ou CSV", 
        type=['xlsx', 'xls', 'csv']
    )
    
    if uploaded_file:
        # Lecture du fichier
        try:
            if uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.success(f"Fichier '{uploaded_file.name}' chargé avec succès !")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            return
        
        # --- 2️⃣ Aperçu des données ---
        st.subheader("Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)

        # --- 3️⃣ Sélection des colonnes ---
        st.subheader("Sélection des colonnes à inclure dans l'étude")
        selected_cols = st.multiselect(
            "Cochez les colonnes à inclure",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
        df_selected = df[selected_cols]
        st.write(f"Colonnes sélectionnées ({len(selected_cols)}): {selected_cols}")

        # --- 4️⃣ Description de l'étude ---
        st.subheader("Décrivez le contexte de votre étude")
        description = st.text_area(
            "Décrivez votre étude en quelques phrases :",
            placeholder="Ex : Étude de l'effet de l'âge et du poids sur la pression artérielle..."
        )
        
        # --- 5️⃣ Extraction de mots-clés ---
        if description:
            tokens = re.findall(r'\b\w+\b', description.lower())
            stopwords_fr = set([
                "le","la","les","un","une","des","de","du","et","en","au","aux","avec",
                "pour","sur","dans","par","au","a","ce","ces","est","sont","ou","où","se",
                "sa","son","que","qui","ne","pas","plus","moins","comme","donc", "d"
            ])
            keywords_fr = [w for w in tokens if w not in stopwords_fr]
            query = " AND ".join(keywords_fr)

            st.write(f"**Mots-clés extraits :** {keywords_fr}")
            st.write(f"**Requête PubMed :** {query}")

            # --- 6️⃣ Recherche PubMed ---
            if st.button("🔍 Rechercher articles PubMed"):
                Entrez.email = "ton.email@example.com"  # à remplacer par ton adresse
                try:
                    handle = Entrez.esearch(db="pubmed", term=query, retmax=10, sort="relevance")
                    record = Entrez.read(handle)
                    handle.close()
                    pmids = record['IdList']
                    
                    if not pmids:
                        st.warning("Aucun article trouvé.")
                    else:
                        st.subheader("Articles PubMed suggérés")
                        for i, pmid in enumerate(pmids, 1):
                            st.markdown(f"{i}. [https://pubmed.ncbi.nlm.nih.gov/{pmid}/](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                except Exception as e:
                    st.error(f"Erreur lors de la recherche PubMed : {e}")
        
        # --- 7️⃣ Récupération du DataFrame sélectionné ---
        st.session_state['df_selected'] = df_selected



