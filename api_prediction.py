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

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não carregado'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            
            required_columns = (preprocessing_info['numeric_features'] + 
                               preprocessing_info['categorical_features'])
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return jsonify({
                    'error': f'Colunas faltantes: {list(missing_columns)}',
                    'required_columns': required_columns
                }), 400
            
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'id': i,
                    'prediction': int(pred),
                    'probability': float(prob),
                    'confidence': 'high' if prob > 0.7 or prob < 0.3 else 'medium'
                })
            
            stats = {
                'total_predictions': len(predictions),
                'positive_predictions': int(np.sum(predictions)),
                'negative_predictions': int(len(predictions) - np.sum(predictions)),
                'average_probability': float(np.mean(probabilities))
            }
            
            return jsonify({
                'predictions': results,
                'statistics': stats,
                'success': True
            })
            
        except Exception as e:
            return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    else:
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
    
@app.route('/predict_single', methods=['POST'])
def predict_single():
    if model is None:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': 'high' if probability > 0.7 or probability < 0.3 else 'medium'
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro na predição: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)