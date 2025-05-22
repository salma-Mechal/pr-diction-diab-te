import joblib
from sklearn.exceptions import NotFittedError
import pandas as pd

# Charger le modèle
model_path = r"c:\Users\pc\OneDrive - Université Sultan Moulay Slimane\Bureau\pfe\modeldiabete3.pkl"
model = joblib.load(model_path)

# Exemple de données d'entrée pour un patient
age = 30
pregnancies = 2
bmi = 25.0
glucose = 120
bloodPressure = 70
ldl = 100
hdl = 50
triglycerides = 150
whr = 0.85
family_history = 1
medication_use = 0
outlier = 0  # ← Ajout de la colonne 'Outlier'

# Les noms de colonnes EXACTEMENT comme pendant l'entraînement du modèle
columns = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 
           'LDL', 'HDL', 'Triglycerides', 'WHR', 'FamilyHistory', 
           'MedicationUse', 'outlier']

# Créer le DataFrame avec toutes les colonnes nécessaires
input_data = pd.DataFrame([[age, pregnancies, bmi, glucose, bloodPressure,
                            ldl, hdl, triglycerides, whr, family_history,
                            medication_use, outlier]], columns=columns)

# Faire la prédiction
try:
    prediction = model.predict(input_data)[0]
    print("Prédiction :", prediction)

    proba = model.predict_proba(input_data)[0][1]
    print("Probabilité d'être diabétique :", round(proba * 100, 2), "%")

except NotFittedError as e:
    print(f"Erreur : {e}")
