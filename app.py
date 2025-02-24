import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
from flask_cors import CORS  # Autoriser les requêtes externes

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = np.ones_like(image) * 255  # Fond blanc
    cv2.drawContours(contour_image, contours, -1, (0, 0, 0), 2)  # Contours noirs

    return contour_image

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

        # Récupérer les seuils de l'utilisateur
        lower_bound = int(request.form.get('lower', 35))
        upper_bound = int(request.form.get('upper', 85))

        processed_image = remove_green_background(image, lower_bound, upper_bound)

        # Sauvegarde temporaire et envoi du fichier
        output_path = "processed.png"
        cv2.imwrite(output_path, processed_image)
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500

@app.route('/detect_contours', methods=['POST'])
def process_contours():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image envoyée"}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Détection des contours
        contour_image = detect_contours(image)

        # Sauvegarde temporaire et envoi du fichier
        output_path = "contours.png"
        cv2.imwrite(output_path, contour_image)
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500

if __name__ == "__main__":
    app.run(debug=True)
