import streamlit as st
from database import *

def show_login():
    st.sidebar.subheader("Connexion")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")
    
    if st.sidebar.button("Se connecter"):
        user_id = verify_user(username, password)
        if user_id:
            st.session_state['user_id'] = user_id
            st.session_state['username'] = username
            st.sidebar.success(f"Connecté en tant que {username}")
        else:
            st.sidebar.error("Identifiants incorrects")

def show_register():
    with st.sidebar.expander("Créer un compte"):
        new_user = st.text_input("Nouvel utilisateur")
        new_pass = st.text_input("Nouveau mot de passe", type="password")
        new_email = st.text_input("Email")
        
        if st.button("S'inscrire"):
            if add_user(new_user, new_pass, new_email):
                st.success("Compte créé avec succès!")
            else:
                st.error("Ce nom d'utilisateur existe déjà")

def show_logout():
    if st.sidebar.button("Se déconnecter"):
        del st.session_state['user_id']
        del st.session_state['username']
        st.experimental_rerun()

def protected_page():
    if 'user_id' not in st.session_state:
        st.warning("Veuillez vous connecter pour accéder à cette page")
        return False
    return True