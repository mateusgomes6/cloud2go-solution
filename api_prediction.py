from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)