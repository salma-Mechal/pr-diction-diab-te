from sklearn.exceptions import NotFittedError
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Diabète",
    page_icon="🧠",
    layout="wide"
)


# Charger le modèle
model_path = r"C:\Users\pc\OneDrive - Université Sultan Moulay Slimane\Bureau\pfe\modeldiabete3.pkl"
model = joblib.load(model_path)

# Charger le CSS
def load_css(file_name):
    with open(file_name) as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("style.css")



st.title("API Web - Prédiction du Risque de Diabète")

tab1, tab2 = st.tabs(["📋 Prédiction manuelle", "📁 Prédictions via fichier"])

# --- Tab 1 : Formulaire manuel ---
with tab1:
    st.subheader("Entrez les données du patient :")
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)
        pregnancies = st.number_input("Grossesses", min_value=0, max_value=20, value=2)
        bmi = st.number_input("IMC (BMI)", min_value=0.0, max_value=70.0, value=25.0)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)

    with col2:
        bloodPressure = st.number_input("Pression Artérielle", min_value=0, max_value=130, value=70)
        ldl = st.number_input("LDL", min_value=0, max_value=300, value=100)
        hdl = st.number_input("HDL", min_value=0, max_value=100, value=50)
        triglycerides = st.number_input("Triglycérides", min_value=0, max_value=500, value=150)

    whr = st.number_input("WHR", min_value=0.0, max_value=2.0, value=0.85)
    family_history = st.number_input("Antécédents familiaux (1=Oui, 0=Non)", min_value=0, max_value=1, value=1)
    medication_use = st.number_input("Médication (1=Oui, 0=Non)", min_value=0, max_value=1, value=0)
    outlier = 0  # valeur par défaut

    if st.button("📌 Prédire"):
        columns = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 
                   'LDL', 'HDL', 'Triglycerides', 'WHR', 'FamilyHistory', 
                   'MedicationUse', 'outlier']
        
        input_data = pd.DataFrame([[age, pregnancies, bmi, glucose, bloodPressure,
                                    ldl, hdl, triglycerides, whr, family_history,
                                    medication_use, outlier]], columns=columns)

        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]

            st.success("Résultat : ✅ Diabétique" if prediction == 1 else "Résultat : ❌ Non diabétique")
            st.info(f"Probabilité : {proba:.2%}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# --- Tab 2 : Upload CSV ou autres formats ---
with tab2:
    st.subheader("Importer un fichier (CSV, Excel, JSON, etc.) :")
    uploaded_file = st.file_uploader("Uploader un fichier", type=["csv", "xlsx","xls", "json"])

    if uploaded_file:
        try:
            # Détecter le type de fichier et charger les données
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_extension == "json":
                df = pd.read_json(uploaded_file)
            else:
                st.error("Format de fichier non pris en charge")
                df = None

            if df is not None:
                st.write("Aperçu des données importées :")
                st.dataframe(df.head())

                # Colonnes attendues selon l'entraînement du modèle
                expected_cols = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 
                                 'LDL', 'HDL', 'Triglycerides', 'WHR', 'FamilyHistory', 
                                 'MedicationUse', 'outlier']

                # Vérifier la présence des colonnes nécessaires
                missing_cols = [col for col in expected_cols if col not in df.columns]
                
                if missing_cols:
                    st.warning(f"Les colonnes suivantes sont manquantes dans votre fichier : {missing_cols}")
                    for col in missing_cols:
                        df[col] = 0  # ou df[col] = np.nan si cela est plus approprié

                # Réorganiser les colonnes pour correspondre à l'ordre attendu par le modèle
                df = df[expected_cols]

                # Prédictions
                preds = model.predict(df)
                probas = model.predict_proba(df)[:, 1]

                df["Résultat"] = ["✅ Diabétique" if p == 1 else "❌ Non diabétique" for p in preds]
                df["Probabilité"] = [f"{p:.2%}" for p in probas]

                st.success("Prédictions réalisées avec succès !")
                st.dataframe(df)

                # Bouton pour télécharger les résultats
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Télécharger les résultats", data=csv, file_name="resultats_diabete.csv", mime='text/csv')

        except Exception as e:
            st.error(f"Erreur lors de la lecture ou la prédiction : {e}")
