import os  # Importe le module pour interagir avec le système de fichiers et les opérations liées aux fichiers.
import cv2  # Importe OpenCV, une bibliothèque pour le traitement d'image.
import numpy as np  # Importe NumPy, utilisé pour les calculs numériques et la manipulation de tableaux.
from flask import Flask, request, jsonify, send_file  # Importe Flask et des fonctions pour gérer les requêtes, répondre avec JSON et envoyer des fichiers.
from flask_cors import CORS  # Importe CORS pour gérer les demandes de ressources cross-origin (partage de ressources entre différentes origines).


app = Flask(__name__)  # Crée une instance de l'application Flask.
CORS(app, resources={r"/*": {"origins": "*"}})  # Active le CORS pour permettre l'accès à l'API depuis n'importe quelle origine.

scale_factor = None  # Variable globale pour stocker l'échelle (mm/pixel)

def remove_green_background(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convertit l'image en espace colorimétrique HSV (teinte, saturation, valeur).
    lower_green = np.array([lower_bound, 50, 50])  # Définit la couleur verte inférieure en HSV avec des bornes ajustées.
    upper_green = np.array([upper_bound, 255, 255])  # Définit la couleur verte supérieure en HSV.
    mask = cv2.inRange(hsv, lower_green, upper_green)  # Crée un masque où les pixels verts sont détectés.
    result = image.copy()  # Crée une copie de l'image originale.
    result[mask == 255] = [255, 255, 255]  # Remplace les pixels verts par du blanc dans l'image.
    return result  # Retourne l'image traitée.

def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertit l'image en niveaux de gris.
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)  # Applique un seuillage pour détecter les objets.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Trouve les contours dans l'image.
    return contours  # Retourne les contours détectés.

def find_closest_contour(contours, click_x, click_y):
    min_distance = float("inf")  # Initialise la distance minimale à l'infini.
    selected_contour = None  # Initialise la variable du contour sélectionné.

    for contour in contours:  # Parcourt chaque contour trouvé.
        for point in contour:  # Parcourt chaque point du contour.
            px, py = point[0]  # Extrait les coordonnées du point.
            distance = np.sqrt((px - click_x) ** 2 + (py - click_y) ** 2)  # Calcule la distance entre le clic et le point.
            if distance < min_distance:  # Si la distance est inférieure à la distance minimale trouvée.
                min_distance = distance  # Met à jour la distance minimale.
                selected_contour = contour  # Met à jour le contour sélectionné.

    return selected_contour  # Retourne le contour le plus proche du clic.

@app.route('/')  # Définit la route pour la page d'accueil.
def home():
    return "Bienvenue sur l'API de traitement d'image !"  # Affiche un message de bienvenue.

@app.route('/process_image', methods=['POST'])  # Définit une route pour traiter les images (méthode POST).
def process_image():
    try:
        if 'image' not in request.files:  # Vérifie si l'image est présente dans la requête.
            return jsonify({"error": "Aucune image envoyée"}), 400  # Si l'image manque, retourne une erreur.

        file = request.files['image']  # Récupère l'image envoyée par l'utilisateur.
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)  # Lit l'image à partir du fichier.

        lower_bound = int(request.form.get('lower', 35))  # Récupère la valeur minimale du seuil pour la couleur verte.
        upper_bound = int(request.form.get('upper', 85))  # Récupère la valeur maximale du seuil pour la couleur verte.

        processed_image = remove_green_background(image, lower_bound, upper_bound)  # Applique la suppression du fond vert.

        output_path = "processed.png"  # Définit le chemin de sortie pour l'image traitée.
        cv2.imwrite(output_path, processed_image)  # Sauvegarde l'image traitée.
        return send_file(output_path, mimetype='image/png')  # Envoie l'image traitée en réponse.

    except Exception as e:  # En cas d'erreur serveur.
        print("Erreur serveur :", str(e))  # Affiche l'erreur sur le serveur.
        return jsonify({"error": "Erreur interne"}), 500  # Retourne une erreur interne au client.

@app.route('/detect_contours', methods=['POST'])  # Définit une route pour détecter les contours (méthode POST).
def detect_contours_api():
    try:
        if 'image' not in request.files:  # Vérifie si l'image est présente dans la requête.
            return jsonify({"error": "Aucune image envoyée"}), 400  # Si l'image manque, retourne une erreur.

        file = request.files['image']  # Récupère l'image envoyée par l'utilisateur.
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)  # Lit l'image à partir du fichier.

        click_x = int(float(request.form.get("click_x", 0)))  # Récupère la coordonnée X du clic.
        click_y = int(float(request.form.get("click_y", 0)))  # Récupère la coordonnée Y du clic.

        contours = detect_contours(image)  # Détecte les contours dans l'image.
        selected_contour = find_closest_contour(contours, click_x, click_y)  # Trouve le contour le plus proche du clic.

        if selected_contour is not None:  # Si un contour est trouvé.
            cv2.drawContours(image, [selected_contour], -1, (255, 0, 0), 2)  # Dessine le contour sélectionné sur l'image.

        output_path = "selected_contour.png"  # Définit le chemin de sortie pour l'image avec le contour.
        cv2.imwrite(output_path, image)  # Sauvegarde l'image avec le contour dessiné.
        return send_file(output_path, mimetype='image/png')  # Envoie l'image avec le contour en réponse.

    except Exception as e:  # En cas d'erreur serveur.
        print("Erreur serveur :", str(e))  # Affiche l'erreur sur le serveur.
        return jsonify({"error": "Erreur interne"}), 500  # Retourne une erreur interne au client.



@app.route('/set_scale', methods=['POST'])
def set_scale():
    """Définit l'échelle de conversion pixels -> mm à partir de deux points donnés."""
    global scale_factor
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image envoyée"}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Récupère les points envoyés par le frontend
        x1, y1 = int(request.form.get("x1", 0)), int(request.form.get("y1", 0))
        x2, y2 = int(request.form.get("x2", 0)), int(request.form.get("y2", 0))
        real_length_mm = float(request.form.get("real_length", 10))  # Longueur réelle en mm

        # Calcule la distance entre les deux points en pixels
        pixel_length = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))

        if pixel_length == 0:
            return jsonify({"error": "Les points doivent être distincts"}), 400

        # Calcule le facteur d'échelle (mm/pixel)
        scale_factor = real_length_mm / pixel_length
        return jsonify({"scale_factor": scale_factor})

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500


@app.route('/calculate_surface', methods=['POST'])
def calculate_surface():
    """Calcule la surface d'un contour sélectionné et la convertit en cm² si l'échelle est définie."""
    global scale_factor
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image envoyée"}), 400

        if scale_factor is None:
            return jsonify({"error": "L'échelle n'est pas encore définie"}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        click_x = int(float(request.form.get("click_x", 0)))
        click_y = int(float(request.form.get("click_y", 0)))

        contours = detect_contours(image)
        selected_contour = find_closest_contour(contours, click_x, click_y)

        if selected_contour is None:
            return jsonify({"error": "Aucun contour trouvé"}), 400

        surface_pixels = cv2.contourArea(selected_contour)  # Surface en pixels²
        surface_cm2 = (surface_pixels * (scale_factor ** 2)) / 100  # Conversion en cm²

        return jsonify({"surface_cm2": surface_cm2})

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500