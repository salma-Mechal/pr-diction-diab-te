import requests

#URL de base de l'API
url_base = "http://127.0.0.1:8000"


#test du endinr d'accueil
response = requests.get(f"{url_base}/")
print("Réponse du endpoint d'accueil:",response.text)

#Données d'exemple pour la prediction
données_predire={
    " pregnancies ": 2,
    "glucose" : 138,
    "bloodPressure" : 62,
    "skinthickness" : 35,
    "insulin" : 0,
    "BMI" : 33.6,
    "DPF ": 0.127,
    "age" : 47,
}

#test du endpoint de prediction 
response = requests.post(f"{url_base}/prediction",json=données_predire)
print("Réponse du endpoint de prediction:",response.text)

#Donées d'exemple pour la prediction avec haute probabilité
données_predire_haute_diabéte ={
    " pregnancies ": 8,
    "glucose" : 180,
    "bloodPressure" : 62,
    "skinthickness" : 35,
    "insulin" : 0,
    "BMI" : 33.6,
    "DPF ": 0.127,
    "age" : 47, 
}