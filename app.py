from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import shutil
import os
from pathlib import Path
from typing import List

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def remove_green_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = image.copy()
    result[mask == 255] = [255, 255, 255]
    return result

def detect_contours(image):
    image = remove_green_background(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def calculate_surface(external_contours, internal_contours, scale_factor):
    external_area = sum(cv2.contourArea(contour) * (scale_factor ** 2) for contour in external_contours)
    internal_area = sum(cv2.contourArea(contour) * (scale_factor ** 2) for contour in internal_contours)
    return external_area - internal_area

@app.post("/calculate_surface/")
async def calculate_surface_api(file: UploadFile = File(...), scale_factor: float = Form(...)):
    file_path = Path(UPLOAD_DIR) / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    image = cv2.imread(str(file_path))
    contours, hierarchy = detect_contours(image)
    
    if hierarchy is None:
        return {"error": "Aucun contour détecté."}
    
    external_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    internal_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] != -1]
    
    surface = calculate_surface(external_contours, internal_contours, scale_factor)
    
    return {"surface_mm2": round(surface, 2)}
