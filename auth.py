import sqlite3
import bcrypt
from database import get_db_connection
from face_auth import get_face_encoding  

def does_user_exist(user_name, user_email):
    """
    Vérifie si l'utilisateur est déjà enregistré dans la base de données.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = ? OR email = ?", (user_name, user_email))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def create_user_account(user_name, user_email, user_password, face_image):
    """
    Ajoute un nouvel utilisateur avec un mot de passe sécurisé et une reconnaissance faciale.
    """
    if does_user_exist(user_name, user_email):
        return "Erreur : L'utilisateur ou l'email existe déjà."
    
    hashed_password = bcrypt.hashpw(user_password.encode('utf-8'), bcrypt.gensalt())
    
    face_encoding = get_face_encoding(face_image)  
    if face_encoding is None:
        return "Erreur lors de l'analyse faciale."
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, email, password, face_encoding) VALUES (?, ?, ?, ?)", 
                   (user_name, user_email, hashed_password.decode('utf-8'), face_encoding.tobytes()))
    conn.commit()
    conn.close()
    
    return f"Utilisateur {user_name} enregistré avec succès."
