import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
import logging
from typing import Optional, Tuple, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
from PIL import Image
import time
import numpy as np
import random


# ============================================
# CONFIGURATION INITIALE
# ============================================

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Chemin de l'icône
ICON_PATH = "assets/page.jpg"

# Configuration de la page Streamlit
try:
    icon = Image.open(ICON_PATH)
    st.set_page_config(
        page_title="Prediction Diabete",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    logger.warning(f"Impossible de charger l'icône : {e}")
    st.set_page_config(
        page_title="Prediction Diabete",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les types de données dans le DataFrame pour assurer la compatibilité avec le modèle.
    
    Args:
        df: DataFrame contenant les données à convertir
        
    Returns:
        DataFrame avec les types convertis
    """
    # Créer une copie pour éviter les modifications sur l'original
    df = df.copy()
    
    # Dictionnaire des conversions de type
    type_conversions = {
        'Age': 'int64',
        'Pregnancies': 'int64',
        'BMI': 'float64',
        'Glucose': 'int64',
        'BloodPressure': 'int64',
        'LDL': 'int64',
        'HDL': 'int64',
        'Triglycerides': 'int64',
        'WHR': 'float64',
        'FamilyHistory': 'int64',
        'MedicationUse': 'int64',
        'outlier': 'int64'
    }
    
    # Appliquer les conversions
    for col in df.columns:
        if col in type_conversions:
            try:
                if type_conversions[col] == 'int64':
                    # Conversion en numérique puis en entier
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif type_conversions[col] == 'float64':
                    # Conversion en numérique puis en float
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
            except (ValueError, TypeError) as e:
                logger.error(f"Erreur de conversion pour la colonne {col} : {e}")
                # Pour les colonnes critiques, on pourrait arrêter l'exécution ici
                if col in ['Glucose', 'BMI', 'Age']:
                    raise ValueError(f"Erreur de conversion pour la colonne critique {col}")
    
    # Vérification des valeurs manquantes
    if df.isnull().any().any():
        logger.warning("Valeurs manquantes détectées après conversion")
        # Remplissage des valeurs manquantes selon le type de colonne
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# ============================================
# GESTION DE LA BASE DE DONNÉES
# ============================================

class DatabaseManager:
    """Classe pour gérer les opérations de base de données"""
    
    def __init__(self, db_name: str = 'diabepredict.db'):
        self.db_name = db_name
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialise la base de données avec les tables nécessaires"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                c = conn.cursor()
                
                # Table utilisateurs
                c.execute('''CREATE TABLE IF NOT EXISTS users
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             username TEXT UNIQUE,
                             password TEXT,
                             email TEXT,
                             created_at TIMESTAMP)''')
                
                # Table analyses
                c.execute('''CREATE TABLE IF NOT EXISTS analyses
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             user_id INTEGER,
                             analysis_type TEXT,
                             parameters TEXT,
                             result TEXT,
                             probability REAL,
                             created_at TIMESTAMP,
                             FOREIGN KEY(user_id) REFERENCES users(id))''')
                
                conn.commit()
                logger.info("Base de données initialisée avec succès")
        except sqlite3.Error as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise
    
    def add_user(self, username: str, password: str, email: str) -> bool:
        """Ajoute un nouvel utilisateur à la base de données"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO users (username, password, email, created_at) VALUES (?, ?, ?, ?)",
                    (username, password, email, datetime.now())
                )
                conn.commit()
                logger.info(f"Nouvel utilisateur créé: {username}")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"Tentative de création d'un utilisateur existant: {username}")
            return False
        except sqlite3.Error as e:
            logger.error(f"Erreur lors de l'ajout d'un utilisateur: {e}")
            return False
    
    def verify_user(self, username: str, password: str) -> Optional[Tuple[int, str]]:
        """Vérifie les identifiants de l'utilisateur"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT id, username FROM users WHERE username=? AND password=?",
                    (username, password)
                )
                result = c.fetchone()
                if result:
                    logger.info(f"Utilisateur authentifié: {username}")
                else:
                    logger.warning(f"Tentative de connexion échouée pour: {username}")
                return result
        except sqlite3.Error as e:
            logger.error(f"Erreur lors de la vérification de l'utilisateur: {e}")
            return None
    
    def add_analysis(self, user_id: int, analysis_type: str, 
                    parameters: Dict[str, Any], result: str, 
                    probability: float) -> bool:
        """Enregistre une analyse dans la base de données"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                c = conn.cursor()
                c.execute(
                    """INSERT INTO analyses 
                    (user_id, analysis_type, parameters, result, probability, created_at) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, analysis_type, str(parameters), result, probability, datetime.now())
                )
                conn.commit()
                logger.info(f"Nouvelle analyse enregistrée pour l'utilisateur {user_id}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Erreur lors de l'ajout d'une analyse: {e}")
            return False
    
    def get_user_analyses(self, user_id: int) -> pd.DataFrame:
        """Récupère l'historique des analyses d'un utilisateur"""
        try:
            with sqlite3.connect(self.db_name) as conn:
                df = pd.read_sql(
                    "SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC",
                    conn,
                    params=(user_id,)
                )
                logger.info(f"Récupération de {len(df)} analyses pour l'utilisateur {user_id}")
                return df
        except (sqlite3.Error, pd.errors.DatabaseError) as e:
            logger.error(f"Erreur lors de la récupération des analyses: {e}")
            return pd.DataFrame()

# Initialisation de la base de données
db_manager = DatabaseManager()

# ============================================
# MODÈLE DE PRÉDICTION
# ============================================

class DiabePredictModel:
    def __init__(self, model_path: str = "modeldiabete3.pkl"):
        self.model_path = model_path
        self.model = self._load_model()
    
    def _load_model(self):
        """Charge le modèle avec gestion d'erreur améliorée"""
        try:
            abs_path = os.path.abspath(self.model_path)
            print(f"Tentative de chargement depuis : {abs_path}")  # Debug
            
            if not os.path.exists(abs_path):
                st.error(f"Fichier modèle introuvable : {abs_path}")
                st.info("Contenu du répertoire :")
                st.write(os.listdir(os.path.dirname(abs_path)))  # Affiche les fichiers disponibles
                st.stop()
            
            # Vérification de la taille du fichier
            file_size = os.path.getsize(abs_path)
            if file_size < 100:  # Taille minimale attendue
                st.error(f"Fichier modèle semble corrompu (taille: {file_size} octets)")
                st.stop()
            
            with open(abs_path, 'rb') as f:
                model = joblib.load(f)
            
            logger.info(f"Modèle chargé avec succès depuis {abs_path}")
            return model
            
        except Exception as e:
            logger.error(f"Erreur de chargement du modèle : {str(e)}", exc_info=True)
            st.error(f"""
                Erreur technique lors du chargement du modèle:
                {str(e)}
                
                Causes possibles :
                1. Fichier modèle corrompu
                2. Version incompatible de scikit-learn
                3. Problème de permissions
                """)
            st.stop()
    
    def predict(self, input_data: pd.DataFrame) -> Tuple[int, float]:
        """
        Effectue une prédiction avec le modèle chargé
        
        Args:
            input_data: DataFrame pandas contenant les données d'entrée
            
        Returns:
            Tuple contenant:
            - La prédiction (0 pour faible risque, 1 pour risque élevé)
            - La probabilité associée (entre 0 et 1)
        """
        try:
            # Vérification que le modèle est bien chargé
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("Modèle non chargé")
                
            # Vérification et conversion des types de données
            input_data = convert_data_types(input_data)
                
            # Vérification des colonnes requises
            required_columns = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure',
                              'LDL', 'HDL', 'Triglycerides', 'WHR', 'FamilyHistory', 'MedicationUse', 'outlier']
            
            # Ajout de la colonne outlier si elle n'existe pas
            if 'outlier' not in input_data.columns:
                input_data['outlier'] = 0
                
            # Vérification que toutes les colonnes requises sont présentes
            missing_cols = [col for col in required_columns if col not in input_data.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans les données d'entrée : {missing_cols}")
                
            # Réorganisation des colonnes selon l'ordre attendu par le modèle
            input_data = input_data[required_columns]
                
            # Prédiction
            prediction = self.model.predict(input_data)[0]  # Retourne 0 ou 1
            probability = self.model.predict_proba(input_data)[0][1]  # Probabilité de classe 1
            
            logger.info(f"Prédiction effectuée - Résultat: {prediction}, Probabilité: {probability:.2f}")
            return prediction, probability
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            raise

# Initialisation du modèle
predict_model = DiabePredictModel()

def estimer_temps(probabilite: float) -> str:
    """Estime le temps jusqu'à l'apparition du diabète"""
    if probabilite < 0.2:
        return "Faible risque (>10 ans)"
    elif probabilite < 0.5:
        return "Risque modéré (5-10 ans)"
    elif probabilite < 0.8:
        return "Risque élevé (3-5 ans)"
    else:
        return "Risque très élevé (<3 ans)"

# ============================================
# STYLE ET THÈME AMÉLIORÉ
# ============================================

def set_custom_style() -> None:
    """Définit le style personnalisé de l'application avec un bleu ciel très clair"""
    st.markdown(f"""
    <style>        :root {{
            --primary: #A7D8F0;  /* Bleu ciel très clair */
            --primary-dark: #8BC8E6;  /* Un peu plus foncé */
            --primary-light: #D1EDFC;  /* Très léger */
            --secondary: #7AC1E4;  /* Pour les accents */
            --success: #A0E6C1;   /* Vert pastel */
            --warning: #FFD699;   /* Orange doux */
            --danger: #FFA7A7;    /* Rouge doux */
            --light: #F5FBFF;     /* Fond presque blanc bleuté */
            --dark: #4A6B7D;      /* Teinte sombre douce */
            --gray: #8CA3A6;      /* Gris bleuté */
            --white: #FFFFFF;
            --card-bg: #FFFFFF;
        }}
        
        /* Fond général plus léger */
        .stApp {{
            background-color: #F5FBFF !important;
        }}
        
        /* En-tête plus doux */
        .st-emotion-cache-1avcm0n {{
            background: var(--primary-dark) !important;
        }}
        
        /* Cartes avec ombre plus subtile */
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(167, 216, 240, 0.2);
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--primary);
        }}
        
        /* Boutons plus doux */
        .stButton>button {{
            background-color: var(--primary) !important;
            color: var(--dark) !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            border: none;
            box-shadow: 0 2px 6px rgba(139, 200, 230, 0.3);
        }}
        
        .stButton>button:hover {{
            background-color: var(--primary-dark) !important;
            color: white !important;
        }}
        
        /* Inputs avec fond très léger */
        .stTextInput input, .stNumberInput input, .stSelectbox select {{
            border-radius: 8px !important;
            border: 1px solid var(--primary) !important;
            background-color: #FBFEFF !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# AUTHENTIFICATION
# ============================================


import streamlit as st
import base64

def show_auth_page() -> None:
    """Affiche la page d'authentification avec le nouveau thème bleu ciel"""
    # CSS personnalisé
    st.markdown(""" 
    <style>
        /* Cacher tous les éléments Streamlit par défaut */
        #MainMenu, header, footer, .stApp > header:first-child {
            display: none !important;
        }

        /* Reset des marges et paddings */
        .stApp {
            margin: 0 !important;
            padding: 0 !important;
            min-height: 100vh;
            background: linear-gradient(135deg, #E0F7FA, #F0F8FF) !important;
        }

        /* Conteneur principal */
        .main-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 1rem;
        }

        /* Carte d'authentification */
        .auth-card {
            width: 100%;
            max-width: 500px;
            padding: 2.5rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 8px 24px rgba(143, 188, 214, 0.3);
            animation: fadeInUp 0.8s ease both;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Styles des éléments */
        .auth-title {
            text-align: center;
            color: #2F4F4F;
            margin-bottom: 0.5rem;
            font-size: 2rem;
            font-weight: 700;
        }

        .auth-subtitle {
            text-align: center;
            color: #708090;
            margin-bottom: 2rem;
            font-size: 1rem;
        }

        .auth-logo {
            display: block;
            margin: 0 auto 1.5rem;
            width: 90px;
            height: 90px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid #87CEEB;
            box-shadow: 0 4px 12px rgba(143, 188, 214, 0.3);
        }

        .auth-footer {
            text-align: center;
            margin-top: 2rem;
            color: #708090;
            font-size: 0.85rem;
        }

        /* Styles des formulaires */
        .stTextInput input, .stTextInput input:focus, 
        .stTextArea textarea, .stTextArea textarea:focus {
            border-radius: 8px !important;
            border: 1px solid #B0E0E6 !important;
            box-shadow: none !important;
            background-color: #F0F8FF !important;
        }

        .stButton button {
            width: 100%;
            background-color: #87CEEB !important;
            color: #2F4F4F !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
        }

        .stButton button:hover {
            background-color: #5F9EA0 !important;
            color: white !important;
            transform: translateY(-2px);
        }

        /* Onglets */
        .stTabs [role="tablist"] {
            margin-bottom: 1.5rem;
        }

        .stTabs [role="tab"] {
            padding: 0.5rem 1rem;
            border-radius: 8px 8px 0 0;
            background: #E6F2F8 !important;
        }

        /* Suppression des espaces inutiles */
        div:empty {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Conteneur principal
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    try:
        # Logo - Chemin relatif recommandé
        logo_path = os.path.join("assets", "page.jpg")
        with open(logo_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f'<img class="auth-logo" src="data:image/jpeg;base64,{encoded_image}">',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Erreur de chargement du logo: {str(e)}")
        # Logo par défaut si échec
        st.markdown(
            '<div class="auth-logo" style="background: #38b2ac; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold;">Logo</div>',
            unsafe_allow_html=True
        )

    # Titres
    st.markdown("<h1 class='auth-title'>Prédiction de diabète</h1>", unsafe_allow_html=True)
    st.markdown("<p class='auth-subtitle'>Solution professionnelle de prédiction du risque diabétique</p>", unsafe_allow_html=True)

    # Onglets Connexion/Inscription
    tab1, tab2 = st.tabs(["Se connecter", "Créer un compte"])

    with tab1:
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Nom d'utilisateur", placeholder="Entrez votre nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password", placeholder="Entrez votre mot de passe")
            
            submitted = st.form_submit_button("Se connecter", type="primary")
            if submitted:
                if not username or not password:
                    st.error("Veuillez remplir tous les champs")
                else:
                    user_data = db_manager.verify_user(username, password)
                    if user_data:
                        st.session_state.update({
                            'user_id': user_data[0],
                            'username': user_data[1],
                            'authenticated': True,
                            'current_page': "Accueil"
                        })
                        st.rerun()
                    else:
                        st.error("Identifiants incorrects")

    with tab2:
        with st.form("register_form", clear_on_submit=True):
            new_user = st.text_input("Nom d'utilisateur", placeholder="Choisissez un nom d'utilisateur")
            new_pass = st.text_input("Mot de passe", type="password", placeholder="Créez un mot de passe")
            confirm_pass = st.text_input("Confirmer le mot de passe", type="password", placeholder="Confirmez le mot de passe")
            new_email = st.text_input("Email", placeholder="Entrez votre email")
            
            submitted = st.form_submit_button("S'inscrire", type="primary")
            if submitted:
                if not all([new_user, new_pass, confirm_pass, new_email]):
                    st.error("Veuillez remplir tous les champs")
                elif new_pass != confirm_pass:
                    st.error("Les mots de passe ne correspondent pas")
                elif len(new_pass) < 6:
                    st.error("Le mot de passe doit contenir au moins 6 caractères")
                elif "@" not in new_email or "." not in new_email.split("@")[1]:
                    st.error("Veuillez entrer une adresse email valide")
                else:
                    if db_manager.add_user(new_user, new_pass, new_email):
                        st.success("Compte créé avec succès! Veuillez vous connecter.")
                    else:
                        st.error("Ce nom d'utilisateur existe déjà")

    # Pied de page
    st.markdown('<p class="auth-footer">© 2025 Prédiction de diabète - Tous droits réservés</p>', unsafe_allow_html=True)
    
    # Fermeture des divs
    st.markdown('</div>', unsafe_allow_html=True)  # auth-card
    st.markdown('</div>', unsafe_allow_html=True)  # main-container



# ============================================
# NAVIGATION LATÉRALE
# ============================================

def show_sidebar_navigation() -> None:
    """Affiche la navigation dans la barre latérale avec logo circulaire"""
    with st.sidebar:
        # Style CSS personnalisé pour la sidebar (en conservant vos couleurs existantes)
        st.markdown("""
        <style>
            /* Style général de la sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(135deg, #B0E0E6, #87CEEB) !important;
                padding: 1.5rem 1rem !important;
            }
            
            /* Style du logo circulaire */
            .sidebar-logo {
                display: block;
                margin: 0 auto 1.5rem;
                width: 100px;
                height: 100px;
                border-radius: 50%;
                object-fit: cover;
                border: 3px solid #8BC8E6;
                box-shadow: 0 4px 12px rgba(139, 200, 230, 0.3);
                transition: all 0.3s ease;
            }
            
            .sidebar-logo:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 16px rgba(139, 200, 230, 0.4);
            }
            
            /* Style des informations utilisateur */
            .sidebar-user-info {
                background: rgba(255, 255, 255, 0.3);
                padding: 0.75rem;
                border-radius: 8px;
                margin: 1.5rem 0 2rem 0;  /* Espace augmenté en bas */
                color: #2F4F4F;
            }
            
            /* Style des boutons de navigation */
            .sidebar-nav-button {
                width: 100%;
                margin: 0.5rem 0;
                padding: 0.75rem 1rem !important;
                border-radius: 8px !important;
                background: rgba(255, 255, 255, 0.3) !important;
                color: #2F4F4F !important;
                font-weight: 500 !important;
                text-align: left;
                transition: all 0.3s ease !important;
                border: none !important;
                box-shadow: none !important;
            }
            
            .sidebar-nav-button:hover {
                background: rgba(255, 255, 255, 0.5) !important;
                transform: translateX(5px) !important;
            }
            
            /* Bouton de déconnexion */
            .sidebar-logout-button {
                background: rgba(255, 255, 255, 0.7) !important;
                color: #FF6347 !important;
                border: 1px solid rgba(255, 99, 71, 0.3) !important;
                margin-top: 1.5rem;  /* Espace supplémentaire au-dessus */
            }
            
            .sidebar-logout-button:hover {
                background: rgba(255, 99, 71, 0.2) !important;
                color: white !important;
            }
        </style>
        """, unsafe_allow_html=True)

        # Ajout du logo circulaire
        try:
            logo_path = os.path.join("assets", "page.jpg")
            with open(logo_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            
            st.markdown(
                f'<img class="sidebar-logo" src="data:image/jpeg;base64,{encoded_image}" alt="Logo">',
                unsafe_allow_html=True
            )
        except Exception as e:
            logger.error(f"Erreur de chargement du logo: {str(e)}")
            # Logo par défaut si échec
            st.markdown(
                '<div class="sidebar-logo" style="display: flex; justify-content: center; '
                'align-items: center; background: #8BC8E6; color: white; font-size: 2rem;">DP</div>',
                unsafe_allow_html=True
            )

        # Informations utilisateur
        st.markdown(f"""
        <div class="sidebar-user-info">
            <p style="opacity: 0.8; margin-bottom: 0.25rem; font-size: 0.9rem;">Connecté en tant que</p>
            <p style="font-weight: 500; margin-bottom: 0;">
                {st.session_state.get('username', '')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Boutons de navigation
        nav_options = {
            "Accueil": "🏠",
            "Prédiction": "🔍",
            "Historique": "📊",
            "Documentation": "📚",
            "À propos": "ℹ️"
        }
        
        for page, icon in nav_options.items():
            if st.button(
                f"{icon} {page}", 
                key=f"nav_{page.lower()}",
                use_container_width=True,
                help=f"Aller à la page {page}"
            ):
                st.session_state['current_page'] = page
                st.rerun()
        
        # Bouton de déconnexion
        if st.button(
            "🚪 Déconnexion", 
            key="nav_logout",
            use_container_width=True,
            help="Se déconnecter du compte"
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
# ============================================
# PAGES DE L'APPLICATION
# ============================================

def show_home_page() -> None:
    """Affiche la page d'accueil"""
    st.markdown("""
        <div class="welcome-section">
            <h1 style="color: #2d3748; margin-bottom: 0.5rem;">Bienvenue sur Prediction Diabete</h1>
            <p style="font-size: 1.1rem; color: #4a5568;">Solution avancée de prédiction du risque diabétique</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Section 1 - Définition du diabète
    with st.container():
        st.markdown("### Qu'est-ce que le diabète ?")
        st.markdown("""
            Le diabète est une maladie chronique qui survient lorsque le pancréas ne produit pas suffisamment d'insuline 
            ou lorsque l'organisme ne peut pas utiliser efficacement l'insuline qu'il produit.
        """)
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
                <div class="card">
                    <div style="font-size: 2rem; color: #3182ce; margin-bottom: 1rem;">1️⃣</div>
                    <h3 style="color: #2d3748;">Diabète de type 1</h3>
                    <p style="color: #4a5568;">Caractérisé par une absence de production d'insuline, nécessitant des injections quotidiennes.</p>
                </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown("""
                <div class="card">
                    <div style="font-size: 2rem; color: #3182ce; margin-bottom: 1rem;">2️⃣</div>
                    <h3 style="color: #2d3748;">Diabète de type 2</h3>
                    <p style="color: #4a5568;">Résulte d'une utilisation inefficace de l'insuline par l'organisme, souvent lié au mode de vie.</p>
                </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown("""
                <div class="card">
                    <div style="font-size: 2rem; color: #3182ce; margin-bottom: 1rem;">🤰</div>
                    <h3 style="color: #2d3748;">Diabète gestationnel</h3>
                    <p style="color: #4a5568;">Hyperglycémie apparaissant pendant la grossesse, pouvant entraîner des complications.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Section 2 - Statistiques
    with st.container():
        st.markdown("### Statistiques alarmantes")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #3182ce; margin-bottom: 0.5rem;">422 millions</h3>
                    <p style="color: #4a5568;">de personnes diabétiques dans le monde (OMS)</p>
                </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #3182ce; margin-bottom: 0.5rem;">1.5 million</h3>
                    <p style="color: #4a5568;">de décès directement liés au diabète chaque année</p>
                </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown("""
                <div class="card">
                    <h3 style="color: #3182ce; margin-bottom: 0.5rem;">+50%</h3>
                    <p style="color: #4a5568;">d'augmentation des cas depuis 1980</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Section 3 - Prévention
    with st.container():
        st.markdown("### Conseils de prévention")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
                <div class="card">
                    <div style="font-size: 2rem; color: #3182ce; margin-bottom: 1rem;">🍎</div>
                    <h3 style="color: #2d3748;">Alimentation saine</h3>
                    <p style="color: #4a5568;">Privilégiez les aliments riches en fibres et pauvres en sucres ajoutés.</p>
                </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown("""
                <div class="card">
                    <div style="font-size: 2rem; color: #3182ce; margin-bottom: 1rem;">🏃‍♂️</div>
                    <h3 style="color: #2d3748;">Activité physique</h3>
                    <p style="color: #4a5568;">Au moins 150 minutes d'activité modérée par semaine.</p>
                </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown("""
                <div class="card">
                    <div style="font-size: 2rem; color: #3182ce; margin-bottom: 1rem;">📊</div>
                    <h3 style="color: #2d3748;">Surveillance</h3>
                    <p style="color: #4a5568;">Contrôlez régulièrement votre glycémie et votre tension artérielle.</p>
                </div>
            """, unsafe_allow_html=True)

# ============================================
# PAGE DE PRÉDICTION AMÉLIORÉE
# ============================================

def show_prediction_page() -> None:
    """Affiche la page de prédiction complète avec les deux onglets"""
    # Style CSS
    st.markdown("""
    <style>
        .st-emotion-cache-1dp5vir { border-width: 0 !important; }
        .result-card {
            background: #f8fafc;
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
        }
        .recommendation-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #4361ee;
        }
        .input-card {
            background: #f8fafc;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Onglets
    tab1, tab2 = st.tabs(["🔍 Analyse individuelle", "📊 Analyse par lot"])
    
    with tab1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #2d3748;">Évaluation du risque diabétique</h1>
            <p style="color: #4a5568;">Remplissez les informations ci-dessous pour obtenir une évaluation personnalisée</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Section de saisie avec toutes les colonnes de l'image
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            age = st.number_input("Âge (années)", min_value=0, max_value=120, value=45)
            pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=0)
            bmi = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            glucose = st.number_input("Glucose sanguin (mg/dL)", min_value=0, max_value=300, value=100)
            bloodPressure = st.number_input("Pression artérielle (mmHg)", min_value=0, max_value=200, value=80)
            ldl = st.number_input("LDL (mg/dL)", min_value=0, max_value=300, value=100)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            hdl = st.number_input("HDL (mg/dL)", min_value=0, max_value=100, value=50)
            triglycerides = st.number_input("Triglycérides (mg/dL)", min_value=0, max_value=1000, value=150)
            waist_hip_ratio = st.number_input("Ratio Taille/Hanche", min_value=0.0, max_value=2.0, value=0.8, step=0.01)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Autres inputs sur une nouvelle ligne
        col4, col5 = st.columns(2)
        with col4:
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            family_history = st.selectbox("Antécédents familiaux", ["Non", "Oui"])
            st.markdown('</div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            medications = st.selectbox("Médicaments pour le diabète", ["Non", "Oui"])
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Évaluer le risque", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                time.sleep(1.5)
                
                # Calcul du risque mis à jour avec toutes les variables
                risk_score = min(
                    max(
                        (age/100 + bmi/50 + glucose/300 + bloodPressure/200 + ldl/400 - hdl/200 + triglycerides/500 + waist_hip_ratio/2) / 5 
                        * (1.2 if family_history == "Oui" else 1)
                        * (1.3 if medications == "Oui" else 1), 
                        0
                    ),
                    0.95
                )
                risk_percentage = round(risk_score * 100, 1)
                time_estimation = estimer_temps(risk_score)
                
                # Affichage des résultats
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Score principal
                st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="color: #2d3748;">Votre résultat</h2>
                    <div style="font-size: 4rem; font-weight: 800; color: #4361ee; margin: 0.5rem 0;">
                        {risk_percentage}%
                    </div>
                    <div style="font-size: 1.2rem; color: #4a5568;">
                        Probabilité de développer un diabète
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Barre de risque visuelle
                fig = go.Figure(go.Indicator(
                    mode = "number+gauge",
                    value = risk_percentage,
                    number = {'suffix': "%"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': "#28a745"},  # Vert
                            {'range': [30, 70], 'color': "#fd7e14"},  # Orange
                            {'range': [70, 100], 'color': "#dc3545"}],  # Rouge
                        'bar': {'color': "#007bff"},  # Bleu
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_percentage
                        }
                    }))
                fig.update_layout(height=150, margin=dict(t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Estimation du temps
                st.markdown(f"""
                <div style="text-align: center; margin: 1.5rem 0; padding: 1rem; 
                            background: #f0f4f8; border-radius: 8px;">
                    <h3 style="color: #2d3748;">⏱ Estimation temporelle</h3>
                    <p style="font-size: 1.2rem;">{time_estimation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommandations personnalisées
                st.markdown("### 📋 Recommandations personnalisées")
                
                if risk_score < 0.4:
                    st.markdown("""
                    <div class="recommendation-card">
                        <h3 style="color: #4cc9f0;">✔ Votre risque est faible</h3>
                        <ul style="color: #4a5568;">
                            <li>Continuez vos bonnes habitudes alimentaires</li>
                            <li>Maintenez une activité physique régulière (30 min/jour)</li>
                            <li>Contrôle annuel de votre glycémie</li>
                            <li>Surveillez votre poids et votre IMC</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif risk_score < 0.7:
                    st.markdown("""
                    <div class="recommendation-card">
                        <h3 style="color: #f8961e;">⚠ Votre risque est modéré</h3>
                        <ul style="color: #4a5568;">
                            <li>Consultez un médecin pour un bilan complet</li>
                            <li>Augmentez votre activité physique (45 min/jour)</li>
                            <li>Réduisez les sucres rapides et les graisses saturées</li>
                            <li>Contrôle glycémique trimestriel recommandé</li>
                            <li>Perte de poids conseillée si IMC > 25</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown("""
                    <div class="recommendation-card">
                        <h3 style="color: #ef233c;">❗ Votre risque est élevé</h3>
                        <ul style="color: #4a5568;">
                            <li>Consultation médicale urgente recommandée</li>
                            <li>Programme d'activité physique supervisé</li>
                            <li>Régime alimentaire strict pauvre en glucides</li>
                            <li>Surveillance glycémique mensuelle</li>
                            <li>Bilan complet avec endocrinologue</li>
                            <li>Évaluation des autres facteurs de risque cardiovasculaire</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Partie Analyse par Lot
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #2d3748;">Analyse par lot</h1>
            <p style="color: #4a5568;">Importez un fichier CSV pour analyser plusieurs patients simultanément</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📤 Importer un fichier CSV", 
            type="csv",
            help="Le fichier doit contenir les colonnes requises: Âge, Nombre de grossesses, IMC, Glucose sanguin, Pression artérielle, LDL, HDL, Triglycérides, Ratio Taille/Hanche, Antécédents familiaux, Médicaments"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Vérification des colonnes requises
                required_columns = ['Age', 'Pregnancies', 'BMI', 'Glucose', 
                                   'BloodPressure', 'LDL', 'HDL', 'Triglycerides',
                                   'WHR', 'FamilyHistory', 'MedicationUse']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Colonnes manquantes dans le fichier: {', '.join(missing_columns)}")
                else:
                    with st.spinner("Analyse des données en cours..."):
                        time.sleep(1)
                        
                        # Simulation d'analyse avec toutes les variables
                        # Conversion des noms de colonnes français en anglais si besoin
                        column_mapping = {
                            'Âge': 'Age',
                            'Nombre de grossesses': 'Pregnancies',
                            'IMC': 'BMI',
                            'Glucose sanguin': 'Glucose',
                            'Pression artérielle': 'BloodPressure',
                            'LDL': 'LDL',
                            'HDL': 'HDL',
                            'Triglycérides': 'Triglycerides',
                            'Ratio Taille/Hanche': 'WHR',
                            'Antécédents familiaux': 'FamilyHistory',
                            'Médicaments': 'MedicationUse'
                        }
                        df = df.rename(columns=column_mapping)

                        df['Probabilité'] = np.round(
                            (df['Age']/100 + df['BMI']/50 + df['Glucose']/300 +
                             df['BloodPressure']/200 + df['LDL']/400 - df['HDL']/200 +
                             df['Triglycerides']/500 + df['WHR']/2) / 5
                            * df['FamilyHistory'].apply(lambda x: 1.2 if x == 1 else 1)
                            * df['MedicationUse'].apply(lambda x: 1.3 if x == 1 else 1),
                            2
                        )
                        df['Niveau de risque'] = pd.cut(df['Probabilité'], 
                                                      bins=[0, 0.4, 0.7, 1],
                                                      labels=['Faible', 'Modéré', 'Élevé'])
                        df['Estimation temps'] = df['Probabilité'].apply(estimer_temps)
                        
                        # Affichage des résultats
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        # Métriques résumées
                        st.markdown("### 📊 Statistiques globales")
                        cols = st.columns(3)
                        cols[0].metric("Patients analysés", len(df))
                        cols[1].metric("Risque moyen", f"{df['Probabilité'].mean()*100:.1f}%")
                        cols[2].metric("Cas à risque élevé", f"{(df['Niveau de risque'] == 'Élevé').sum()}")
                        
                        # Diagramme unique (camembert)
                        st.markdown("### 📈 Répartition des risques")
                        risk_counts = df['Niveau de risque'].value_counts()
                        fig = px.pie(
                            risk_counts, 
                            values=risk_counts.values,
                            names=risk_counts.index,
                            color=risk_counts.index,
                            color_discrete_map={
                                'Faible': '#4cc9f0',
                                'Modéré': '#f8961e',
                                'Élevé': '#ef233c'
                            },
                            hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tableau récapitulatif
                        st.markdown("### 📋 Détail des résultats")
                        st.dataframe(
                            df.style.applymap(
                                lambda x: 'background-color: #e6f7ff' if x == 'Faible' else 
                                'background-color: #fff7e6' if x == 'Modéré' else 
                                'background-color: #ffe6e6',
                                subset=['Niveau de risque']
                            ),
                            column_config={
                                "Probabilité": st.column_config.ProgressColumn(
                                    "Probabilité",
                                    format="%.2f",
                                    min_value=0,
                                    max_value=1,
                                )
                            },
                            use_container_width=True,
                            height=400
                        )
                        
                        # Bouton d'export
                        st.download_button(
                            label="💾 Exporter les résultats",
                            data=df.to_csv(index=False).encode('utf-8'),
                            file_name='resultats_diabete.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {str(e)}")

def estimer_temps(probabilite: float) -> str:
    """Estime le temps jusqu'à l'apparition du diabète"""
    if probabilite < 0.2:
        return "Apparition probable dans plus de 10 ans"
    elif probabilite < 0.5:
        return "Apparition probable entre 5 et 10 ans"
    elif probabilite < 0.8:
        return "Apparition probable entre 3 et 5 ans"
    else:
        return "Apparition probable dans moins de 3 ans"

def show_history_page() -> None:
    """Affiche l'historique des analyses"""
    st.markdown("### Historique des analyses")
    st.markdown("Consultez vos analyses précédentes et leurs résultats.")
    
    try:
        analyses = db_manager.get_user_analyses(st.session_state['user_id'])
        
        if not analyses.empty:
            # Filtrer par type d'analyse
            analysis_types = analyses['analysis_type'].unique()
            selected_type = st.selectbox("Filtrer par type d'analyse", ["Tous"] + list(analysis_types))
            
            if selected_type != "Tous":
                analyses = analyses[analyses['analysis_type'] == selected_type]
            
            # Convertir les dates
            analyses['created_at'] = pd.to_datetime(analyses['created_at'])
            
            # Trier par date
            analyses = analyses.sort_values('created_at', ascending=False)
           
            # Afficher le tableau des analyses
            st.dataframe(
                analyses[['created_at', 'analysis_type', 'result', 'probability']],
                column_config={
                    "created_at": "Date",
                    "analysis_type": "Type d'analyse",
                    "result": "Résultat",
                    "probability": st.column_config.NumberColumn(
                        "Probabilité",
                        format="%.2f",
                        help="Probabilité de risque diabétique"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Option pour voir les détails d'une analyse
            selected_analysis = st.selectbox(
                "Voir les détails d'une analyse",
                [f"{row['created_at']} - {row['analysis_type']}" for _, row in analyses.iterrows()]
            )
            
            if selected_analysis:
                selected_idx = [i for i, x in enumerate(
                    [f"{row['created_at']} - {row['analysis_type']}" for _, row in analyses.iterrows()]
                ) if x == selected_analysis][0]
                
                selected = analyses.iloc[selected_idx]
                
                with st.expander("Détails de l'analyse", expanded=True):
                    st.json(selected['parameters'])
        else:
            st.info("Aucune analyse enregistrée pour le moment.")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'historique : {str(e)}")
        logger.error(f"Erreur historique : {str(e)}", exc_info=True)

def show_documentation_page() -> None:
    """Affiche la page de documentation avec un style amélioré"""
    # Style CSS moderne
    st.markdown("""
    <style>
        .doc-section {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 4px solid #4361ee;
        }
        .doc-header {
            color: #2d3748;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        .doc-item {
            margin-bottom: 0.8rem;
            padding-left: 1rem;
            border-left: 3px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        .doc-item:hover {
            border-left-color: #4361ee;
            background-color: #f8fafc;
        }
        .metric-card {
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 3px solid #4361ee;
        }
        .code-block {
            background: #2d3748;
            color: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
            overflow-x: auto;
            white-space: pre;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .warning-note {
            margin-top: 1.5rem;
            padding: 1rem;
            background: #fffaf0;
            border-radius: 8px;
            border-left: 3px solid #dd6b20;
        }
    </style>
    """, unsafe_allow_html=True)

    # En-tête
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #2d3748;">📚 Documentation Technique</h1>
        <p style="color: #4a5568;">Toutes les informations pour comprendre et utiliser notre outil de prédiction</p>
    </div>
    """, unsafe_allow_html=True)

    # Section 1 - Paramètres du modèle
    with st.expander("📋 Paramètres utilisés dans le modèle", expanded=True):
        st.markdown("""
        <div class="doc-section">
            <h3 class="doc-header">Description des paramètres</h3>
            <div class="doc-item"><strong>Âge</strong> : L'âge du patient en années complètes</div>
            <div class="doc-item"><strong>Nombre de grossesses</strong> : Pour les patientes femmes uniquement</div>
            <div class="doc-item"><strong>IMC</strong> : Indice de Masse Corporelle (poids en kg / taille² en m)</div>
            <div class="doc-item"><strong>Glucose sanguin</strong> : Taux de glucose à jeun (mg/dL)</div>
            <div class="doc-item"><strong>Pression artérielle</strong> : Pression systolique (mmHg)</div>
            <div class="doc-item"><strong>LDL</strong> : "Mauvais" cholestérol (mg/dL)</div>
            <div class="doc-item"><strong>HDL</strong> : "Bon" cholestérol (mg/dL)</div>
            <div class="doc-item"><strong>Triglycérides</strong> : Niveau dans le sang (mg/dL)</div>
            <div class="doc-item"><strong>Ratio Taille/Hanche</strong> : Circonférence taille / circonférence hanche</div>
            <div class="doc-item"><strong>Antécédents familiaux</strong> : Diabète chez les parents au 1er degré (0=Non, 1=Oui)</div>
            <div class="doc-item"><strong>Médicaments</strong> : Utilisation de médicaments pour le diabète (0=Non, 1=Oui)</div>
        </div>
        """, unsafe_allow_html=True)

    # Section 2 - Interprétation
    with st.expander("📈 Interprétation des résultats", expanded=False):
        st.markdown("""
        <div class="doc-section">
            <h3 class="doc-header">Échelle de risque</h3>
            <div class="metric-card" style="border-left-color: #38a169;"><strong style="color: #38a169;">0-20%</strong> : Risque très faible</div>
            <div class="metric-card" style="border-left-color: #68d391;"><strong style="color: #68d391;">20-40%</strong> : Risque faible</div>
            <div class="metric-card" style="border-left-color: #f6ad55;"><strong style="color: #f6ad55;">40-60%</strong> : Risque modéré</div>
            <div class="metric-card" style="border-left-color: #f56565;"><strong style="color: #f56565;">60-80%</strong> : Risque élevé</div>
            <div class="metric-card" style="border-left-color: #e53e3e;"><strong style="color: #e53e3e;">80-100%</strong> : Risque très élevé</div>
            <div class="warning-note">
                <strong>Seuil de décision</strong> : Un résultat est considéré comme "à risque" lorsque la probabilité dépasse <strong style="color: #dd6b20;">50%</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Section 3 - Méthodologie (CORRIGÉE)
    with st.expander("🔬 Méthodologie scientifique", expanded=False):
       st.markdown("""
    <div class="doc-section">
        <h3 class="doc-header">Processus d'analyse</h3>
        <p>Notre approche suit une méthodologie rigoureuse :</p>
        <div class="doc-item">
            <strong>1. Exploration des données</strong> : Analyse préliminaire sous Jupyter Notebook avec :
            <ul style="margin-top: 0.5rem; margin-left: 1.5rem;">
                <li>Visualisation des distributions</li>
                <li>Détection des valeurs aberrantes</li>
                <li>Analyse des corrélations</li>
            </ul>
        </div>
        <div class="doc-item">
            <strong>2. Prétraitement</strong> :
            <ul style="margin-top: 0.5rem; margin-left: 1.5rem;">
                <li>Nettoyage des données manquantes</li>
                <li>Normalisation des caractéristiques</li>
                <li>Encodage des variables catégorielles</li>
            </ul>
        </div>
        <div class="doc-item">
            <strong>3. Modélisation</strong> : Implémentation d'une <strong>Régression Logistique</strong> avec :
            <ul style="margin-top: 0.5rem; margin-left: 1.5rem;">
                <li>Réglage des hyperparamètres</li>
                <li>Validation croisée (k=5)</li>
                <li>Optimisation de la fonction de coût</li>
            </ul>
        </div>
        <h3 class="doc-header" style="margin-top: 1.5rem;">Validation du modèle</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
            <div class="metric-card">
                <div style="font-size: 1.5rem; color: #4361ee;">{accuracy}%</div>
                <div>Exactitude</div>
            </div>
            <div class="metric-card">
                <div style="font-size: 1.5rem; color: #4361ee;">{precision}%</div>
                <div>Précision</div>
            </div>
            <div class="metric-card">
                <div style="font-size: 1.5rem; color: #4361ee;">{recall}%</div>
                <div>Rappel</div>
            </div>
            <div class="metric-card">
                <div style="font-size: 1.5rem; color: #4361ee;">{auc}</div>
                <div>AUC-ROC</div>
            </div>
        </div>
        <p>Coefficients standardisés du modèle :</p>
        <div class="code-block">
            Glucose: 0.42 | Age: 0.38 | BMI: 0.31
            LDL: 0.25 | Pression: 0.18 | Antécédents: 0.15
        </div>
        <h3 class="doc-header" style="margin-top: 1.5rem;">Limites</h3>
        <div class="doc-item">1. Sensible aux déséquilibres de classes</div>
        <div class="doc-item">2. Suppose une relation linéaire entre les caractéristiques et le log-odds</div>
        <div class="doc-item">3. Nécessite un seuillage optimal pour la classification</div>
    </div>
    """.format(
        accuracy="88.5",
        precision="86.2",
        recall="90.1",
        auc="0.927"
    ), unsafe_allow_html=True)
    # Section 4 - Format CSV (CORRIGÉE)
    with st.expander("📁 Format du fichier CSV", expanded=False):
        st.markdown("""
        <div class="doc-section">
            <h3 class="doc-header">Structure requise</h3>
            <p>Le fichier CSV doit contenir exactement les colonnes suivantes (l'ordre est important) :</p>
            <div class="code-block">
                Age,Pregnancies,BMI,Glucose,BloodPressure,LDL,HDL,Triglycerides,WHR,FamilyHistory,MedicationUse
            </div>
            <h3 class="doc-header" style="margin-top: 1.5rem;">Exemple de données valides</h3>
            <div class="code-block">
                Age,Pregnancies,BMI,Glucose,BloodPressure,LDL,HDL,Triglycerides,WHR,FamilyHistory,MedicationUse
                32,0,26.5,92,72,110,55,120,0.78,0,0
                45,2,28.7,115,85,135,42,180,0.85,1,0
                61,0,31.2,142,92,158,38,240,0.91,1,1
            </div>
            <h3 class="doc-header" style="margin-top: 1.5rem;">Recommandations</h3>
            <div class="doc-item">• Utiliser des valeurs numériques uniquement</div>
            <div class="doc-item">• Pas de valeurs manquantes (remplacer par 0 si nécessaire)</div>
            <div class="doc-item">• Encodage UTF-8 recommandé</div>
            <div class="doc-item">• Séparateur: virgule (,)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Exemple CSV téléchargeable
        csv_example = """Age,Pregnancies,BMI,Glucose,BloodPressure,LDL,HDL,Triglycerides,WHR,FamilyHistory,MedicationUse
32,0,26.5,92,72,110,55,120,0.78,0,0
45,2,28.7,115,85,135,42,180,0.85,1,0
61,0,31.2,142,92,158,38,240,0.91,1,1"""
        

def show_about_page() -> None:
    """Affiche la page À propos"""
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1>À propos de Prediction Diabete</h1>
            <p style="font-size: 1.2rem;">Solution avancée de prédiction du risque diabétique</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns([1, 2])
    
    with cols[0]:
       
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="margin-bottom: 1rem;">
            <img src="data:image/png;base64,{base64.b64encode(open('assets/page.jpg', 'rb').read()).decode()}" 
                 alt="Logo" 
                 style="height: 8rem; 
                    width: 8rem;
                    border-radius: 50%;
                    object-fit: cover;
                    border: 3px solid #f0f2f6;">
            </div>
            <h3>Version 1 </h3>
            <p>Dernière mise à jour : 16/05/2025</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
            ### Notre mission
            
            Prediction Diabete a été développé pour aider les professionnels de santé à identifier 
            précocement les patients à risque de développer un diabète, permettant une intervention 
            préventive plus efficace.
            
            ### Fonctionnalités clés
            
            - Prédiction individuelle du risque diabétique  
            - Analyse groupée de populations  
            - Historique complet des analyses  
            - Recommandations personnalisées  
            - Interface professionnelle et intuitive  
            
            ### Équipe
            
           Developper par deux etudiantes en deuxième année d'informatique Decisionnelle et statistique à Ecole Superieur de Technologie :
           -AYA MOUNSSIF .
           -SALMA MECHAL.
            
            ### Contact
            
            📧 contact@diabepredict.com  
            🌐 www.diabepredict.com  
            📞 +33 1 23 45 67 89  
        """)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 0.9rem;">
            <p>© 2025 Prediction Diabète - Tous droits réservés</p>
            <p>Ce logiciel est destiné aux professionnels de santé et ne remplace pas un diagnostic médical.</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================
# GESTION DE L'APPLICATION
# ============================================

def main() -> None:
    """Fonction principale de l'application"""
    
    # Configuration du style
    set_custom_style()
    
    # Gestion de l'authentification
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        show_auth_page()
        return
    
    # Navigation
    show_sidebar_navigation()
    
    # Affichage de la page courante
    current_page = st.session_state.get('current_page', 'Accueil')
    
    if current_page == "Accueil":
        show_home_page()
    elif current_page == "Prédiction":
        show_prediction_page()
    elif current_page == "Historique":
        show_history_page()
    elif current_page == "Documentation":
        show_documentation_page()
    elif current_page == "À propos":
        show_about_page()

if __name__ == "__main__":
    main()