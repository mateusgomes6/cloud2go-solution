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