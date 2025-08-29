from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_PATH = 'modelo_treinado.pkl'
PREPROCESSING_INFO_PATH = 'preprocessing_info.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    model = joblib.load(MODEL_PATH)
    with open(PREPROCESSING_INFO_PATH, 'r') as f:
        preprocessing_info = json.load(f)
    print("Modelo e informações carregados com sucesso!")
except FileNotFoundError:
    print("Arquivos do modelo não encontrados. Execute o treinamento primeiro.")
    model = None
    preprocessing_info = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})