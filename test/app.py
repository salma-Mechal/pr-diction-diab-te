import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib

# Charger le modèle
model_path = r"C:\Users\pc\OneDrive - Université Sultan Moulay Slimane\Bureau\pfe\model_diabete2.pkl"
model = joblib.load(model_path)

# Vérifiez le type de modèle pour le débogage
print(type(model))

st.set_page_config(page_title="Prédiction Diabète", page_icon="🧠")
st.title("API Web - Prédiction du Risque de Diabète")

tab1, tab2 = st.tabs(["📋 Prédiction manuelle", "📁 Prédictions via CSV"])

# --- Tab 1 : Formulaire manuel ---
with tab1:
    st.subheader("Entrez les données du patient :")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
    bloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=130, value=70)
    skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    if st.button("📌 Prédire"):
        input_data = np.array([[pregnancies, glucose, bloodPressure, skinthickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.success("Résultat : ✅ Diabétique" if prediction == 1 else "Résultat : ❌ Non diabétique")
        st.info(f"Probabilité : {proba:.2%}")

# --- Tab 2 : Upload CSV ---
with tab2:
    st.subheader("Importer un fichier CSV :")
    uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            st.write("Aperçu des données importées :")
            st.dataframe(df.head())

            expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            if all(col in df.columns for col in expected_cols):
                preds = model.predict(df[expected_cols])
                probas = model.predict_proba(df[expected_cols])[:, 1]

                df["Résultat"] = ["✅ Diabétique" if p == 1 else "❌ Non diabétique" for p in preds]
                df["Probabilité"] = [f"{p:.2%}" for p in probas]

                st.success("Prédictions réalisées avec succès !")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Télécharger les résultats", data=csv, file_name="resultats_diabete.csv", mime='text/csv')

            else:
                st.error("❗ Le fichier ne contient pas toutes les colonnes nécessaires.")

        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
