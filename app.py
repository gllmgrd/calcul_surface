import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

selected_external_contours = []
selected_internal_contours = []

def remove_green_background(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([lower_bound, 50, 50])
    upper_green = np.array([upper_bound, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = image.copy()
    result[mask == 255] = [255, 255, 255]
    return result

def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy

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

        lower_bound = int(request.form.get('lower', 35))
        upper_bound = int(request.form.get('upper', 85))

        processed_image = remove_green_background(image, lower_bound, upper_bound)

        output_path = "processed.png"
        cv2.imwrite(output_path, processed_image)
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500

@app.route('/detect_contours', methods=['POST'])
def detect_contours_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image envoyée"}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        contours, hierarchy = detect_contours(image)

        if hierarchy is None:
            return jsonify({"error": "Aucun contour détecté"}), 400

        external_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
        internal_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] != -1]

        return jsonify({
            "external_contours": [c.tolist() for c in external_contours],
            "internal_contours": [c.tolist() for c in internal_contours]
        })

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500

@app.route('/select_contours', methods=['POST'])
def select_contours():
    try:
        data = request.get_json()
        contour_type = data.get("type")  # "external" ou "internal"
        selected_contours = data.get("contours")

        if contour_type == "external":
            global selected_external_contours
            selected_external_contours = selected_contours
        elif contour_type == "internal":
            global selected_internal_contours
            selected_internal_contours = selected_contours

        return jsonify({"message": f"Contours {contour_type} sélectionnés."}), 200

    except Exception as e:
        print("Erreur serveur :", str(e))
        return jsonify({"error": "Erreur interne"}), 500

if __name__ == "__main__":
    app.run(debug=True)