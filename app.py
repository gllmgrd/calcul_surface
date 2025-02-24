import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
from flask_cors import CORS  # Autoriser les requêtes depuis Squarespace

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Autorise toutes les origines

# Fonction pour supprimer le fond vert
def remove_green_background(image, lower_bound, upper_bound):
    """Supprime le fond vert avec seuils ajustables."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([lower_bound, 50, 50])
    upper_green = np.array([upper_bound, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = image.copy()
    result[mask == 255] = [255, 255, 255]  # Fond blanc

    return result


# Fonction pour détecter les contours
def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


# Fonction pour calculer la surface
def calculate_surface(external_contours, internal_contours, scale_factor):
    external_area = sum(cv2.contourArea(contour) * (scale_factor ** 2) for contour in external_contours)
    internal_area = sum(cv2.contourArea(contour) * (scale_factor ** 2) for contour in internal_contours)
    total_area = external_area - internal_area
    return total_area

@app.route('/')
def home():
    return "Bienvenue sur l'API de traitement d'image !"

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image envoyée"}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Récupérer les seuils de l'utilisateur (avec valeurs par défaut)
        lower_bound = int(request.form.get('lower', 35))
        upper_bound = int(request.form.get('upper', 85))

        processed_image = remove_green_background(image, lower_bound, upper_bound)

        # Sauvegarde temporaire et envoi du fichier
        output_path = "processed.png"
        cv2.imwrite(output_path, processed_image)
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
            print("Erreur serveur :", str(e))  # Ajout d'un log dans Render
            return jsonify({"error": "Erreur interne"}), 500

@app.route('/detect_contours', methods=['POST'])
def detect_contours_api():
    try:
        data = request.get_json()
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({"error": "Aucune image reçue"}), 400

        # Télécharger l'image depuis l'URL temporaire
        image = Image.open(io.BytesIO(requests.get(image_url).content))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Détection des contours
        contours, hierarchy = detect_contours(image)
        cv2.drawContours(image, contours, -1, (255, 0, 0), 2)  # Dessiner les contours en bleu

        # Sauvegarde et envoi du fichier
        output_path = "contours.png"
        cv2.imwrite(output_path, image)
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500

if __name__ == "__main__":
    app.run(debug=True)