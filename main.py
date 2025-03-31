import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import dlib
from database import get_db_connection
import bcrypt

# Configuration initiale
st.set_page_config(page_title="Connexion Faciale", layout="centered")
detecteur_visage = dlib.get_frontal_face_detector()
marqueurs_faciaux = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Capture d'une image unique
def prendre_photo():
    camera = cv2.VideoCapture(0)
    ret, image = camera.read()
    camera.release()
    
    if not ret:
        st.error("Échec de la capture d'image")
        return None
    
    st.image(image, channels="BGR", caption="Image capturée", use_container_width=True)
    return image

# Vérifier l'existence de l'utilisateur
def utilisateur_existe(nom, courriel):
    connexion =get_db_connection()
    curseur = connexion.cursor()
    curseur.execute("SELECT 1 FROM users WHERE username = ? OR email = ?", (nom, courriel))
    resultat = curseur.fetchone()
    connexion.close()
    return resultat is not None

# Extraire les caractéristiques faciales
def extraire_marqueurs(image):
    image_grise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visages = detecteur_visage(image_grise)
    if len(visages) == 0:
        st.error("Aucun visage détecté.")
        return None
    
    marqueurs = marqueurs_faciaux(image_grise, visages[0])
    return np.array([marqueurs.part(i).x for i in range(68)] + [marqueurs.part(i).y for i in range(68)])

# Inscription d'un utilisateur
def inscrire_utilisateur(nom, courriel, mot_de_passe, photo):
    if utilisateur_existe(nom, courriel):
        st.error("Cet utilisateur existe déjà.")
        return
    
    mdp_hache = bcrypt.hashpw(mot_de_passe.encode(), bcrypt.gensalt())
    encodage_visage = extraire_marqueurs(photo)
    if encodage_visage is None:
        return
    
    connexion = get_db_connection()
    curseur = connexion.cursor()
    curseur.execute("INSERT INTO users (username, email, password, face_encoding) VALUES (?, ?, ?, ?)", 
                   (nom, courriel, mdp_hache.decode(), encodage_visage.tobytes()))
    connexion.commit()
    connexion.close()
    st.success(f"Utilisateur {nom} inscrit avec succès.")

# Comparer les visages
def verifier_identite(image_capturee, encodage_stocke, seuil=100):
    encodage_capture = extraire_marqueurs(image_capturee)
    if encodage_capture is None:
        return False
    
    distance = np.linalg.norm(encodage_capture - np.frombuffer(encodage_stocke, dtype=np.float64))
    return distance < seuil

# Connexion utilisateur
def authentifier_utilisateur(nom, mot_de_passe):
    photo_capturee = prendre_photo()
    if photo_capturee is None:
        return "Erreur lors de la capture."
    
    connexion =get_db_connection()
    curseur = connexion.cursor()
    curseur.execute("SELECT password, face_encoding FROM users WHERE username = ?", (nom,))
    donnees = curseur.fetchone()
    connexion.close()
    
    if donnees and bcrypt.checkpw(mot_de_passe.encode(), donnees[0].encode()):
        if verifier_identite(photo_capturee, donnees[1], 5000):
            st.session_state.utilisateur_connecte = True
            return "Connexion réussie !"
        return "Échec de la reconnaissance faciale."
    return "Identifiants incorrects."

# Interface Streamlit
def interface():
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: black; }
        .stButton button { background-color: #4CAF50; color: white; }
        h1, h2, h3, h4, h5, h6, p, label, div { color: black !important; }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🔐 Accès par Reconnaissance Faciale")
    if 'utilisateur_connecte' not in st.session_state:
        st.session_state.utilisateur_connecte = False
    
    menu = st.sidebar.selectbox("Navigation", ["Créer un compte", "Se connecter"] if not st.session_state.utilisateur_connecte else ["Accueil", "Se déconnecter"])
    
    if menu == "Créer un compte":
        st.subheader("Inscription")
        nom = st.text_input("Nom d'utilisateur")
        courriel = st.text_input("Email")
        mot_de_passe = st.text_input("Mot de passe", type='password')
        if st.button("Capturer & Enregistrer"):
            if nom and courriel and mot_de_passe:
                photo = prendre_photo()
                if photo is not None:
                    inscrire_utilisateur(nom, courriel, mot_de_passe, photo)
            else:
                st.warning("Veuillez remplir tous les champs.")
    
    elif menu == "Se connecter":
        st.subheader("Connexion")
        nom = st.text_input("Nom d'utilisateur")
        mot_de_passe = st.text_input("Mot de passe", type='password')
        if st.button("Se connecter"):
            if nom and mot_de_passe:
                st.write(authentifier_utilisateur(nom, mot_de_passe))
            else:
                st.warning("Veuillez renseigner vos identifiants.")
    
    elif menu == "Accueil":
        st.success("Bienvenue !")
    
    elif menu == "Se déconnecter":
        st.session_state.utilisateur_connecte = False
        st.success("Déconnexion réussie.")

if __name__ == "__main__":
    interface()
