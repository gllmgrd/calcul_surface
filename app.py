from flask import Flask, request, jsonify
import cv2
import numpy as np

import os
port = int(os.environ.get('PORT', 5000))
app = Flask(__name__)


def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


@app.route('/detect_contours', methods=['POST'])
def detect_contours_api():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    contours = detect_contours(image)

    return jsonify({"message": "Traitement terminé", "contours_detectes": len(contours)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)