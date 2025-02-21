import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS  # Autoriser les requêtes depuis Squarespace

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Autorise toutes les origines

# Fonction pour supprimer le fond vert
def remove_green_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound, upper_bound = 35, 85  # Valeurs initiales
    lower_green = np.array([lower_bound, 50, 50])
    upper_green = np.array([upper_bound, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = image.copy()
    result[mask == 255] = [255, 255, 255]
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
            return jsonify({"error": "Aucune image reçue"}), 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = np.array(image)
        contours, hierarchy = detect_contours(image)

        # Appel à d'autres fonctions comme `select_contours` et `calculate_surface` ici

        # Simulation d'une surface calculée
        surface = 123.45  # Remplacer par ton calcul réel

        return jsonify({'surface': surface})

    except Exception as e:
            print("Erreur serveur :", str(e))  # Ajout d'un log dans Render
            return jsonify({"error": "Erreur interne"}), 500

if __name__ == "__main__":
    app.run(debug=True)
