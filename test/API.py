import streamlit as st
import joblib
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Fonction pour entraîner et sauvegarder un modèle si besoin
def train_and_save_model():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = (data.target > data.target.mean()).astype(int)  # Binarisation cible
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'xgb_diabetes_model.pkl')
    return model

# Charger le modèle ou l'entraîner si non existant
try:
    model = joblib.load('xgb_diabetes_model.pkl')
except:
    model = train_and_save_model()

st.title("Prédiction du diabète avec XGBoost")

st.write("""
Entrez les caractéristiques du patient pour prédire la présence ou non du diabète.
""")

# Créer les champs d'entrée utilisateur
def user_input_features():
    data = load_diabetes()
    inputs = {}
    for feature in data.feature_names:
        # On propose une valeur moyenne comme valeur par défaut
        mean_val = np.mean(data.data[:, data.feature_names.index(feature)])
        inputs[feature] = st.number_input(f"{feature}", value=float(mean_val))
    return pd.DataFrame([inputs])

# Obtenir les inputs utilisateur
input_df = user_input_features()

# Bouton pour prédire
if st.button("Prédire"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    
    st.write(f"**Prédiction :** {'Diabétique' if prediction[0] == 1 else 'Non diabétique'}")
    st.write(f"**Probabilité (Diabétique) :** {proba[0][1]*100:.2f}%")

