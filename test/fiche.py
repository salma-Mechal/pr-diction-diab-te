import pandas as pd
import numpy as np

# Définir les colonnes demandées
columns = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 
           'LDL', 'HDL', 'Triglycerides', 'WHR', 'FamilyHistory', 
           'MedicationUse']

# Générer des données aléatoires réalistes
num_rows = 100  # Nombre de lignes

data = {
    'Age': np.random.randint(18, 80, num_rows),
    'Pregnancies': np.random.randint(0, 10, num_rows),
    'BMI': np.round(np.random.uniform(18.5, 40, num_rows), 1),
    'Glucose': np.random.randint(70, 200, num_rows),
    'BloodPressure': np.random.randint(60, 180, num_rows),
    'LDL': np.random.randint(50, 200, num_rows),
    'HDL': np.random.randint(20, 100, num_rows),
    'Triglycerides': np.random.randint(50, 500, num_rows),
    'WHR': np.round(np.random.uniform(0.7, 1.2, num_rows), 2),
    'FamilyHistory': np.random.choice(['1', '0'], num_rows),
    'MedicationUse': np.random.choice(['1', '0'], num_rows)
}

# Créer un DataFrame
df_generated = pd.DataFrame(data)

# Détecter les outliers (valeurs extrêmes)
def detect_outlier(row):
    if (row['Glucose'] > 180 or row['BMI'] > 35 or row['BloodPressure'] > 140 or 
        row['LDL'] > 160 or row['HDL'] < 30 or row['Triglycerides'] > 400):
        return '1'
    return '0'

df_generated['outlier'] = df_generated.apply(detect_outlier, axis=1)

# Sauvegarder en CSV
df_generated.to_csv("donnees_generees.csv", index=False, encoding="utf-8")

print("Fichier CSV généré avec colonne 'Outlier' : donnees_generees.csv")


df_generated.to_csv("C:\\Users\\pc\\OneDrive - Université Sultan Moulay Slimane\\Bureau\\pfe\\test\\donnees_generees.csv", index=False, encoding="utf-8")


