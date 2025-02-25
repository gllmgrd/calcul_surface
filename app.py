import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

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
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

@app.route('/detect_contours', methods=['POST'])
def detect_contours_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image envoyée"}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        contours = detect_contours(image)

        return jsonify({"contours": list(range(len(contours)))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/apply_contour', methods=['POST'])
def apply_contour():
    try:
        if 'image' not in request.files or 'contour_index' not in request.form:
            return jsonify({"error": "Image ou index manquant"}), 400

        file = request.files['image']
        contour_index = int(request.form['contour_index'])

        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        contours = detect_contours(image)

        if contour_index == -1:
            cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
        else:
            cv2.drawContours(image, [contours[contour_index]], -1, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.png', image)
        return send_file(io.BytesIO(buffer), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)