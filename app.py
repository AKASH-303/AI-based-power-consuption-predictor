from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
from pathlib import Path
import joblib
import os
import pandas as pd
from model import ensure_model, predict_single, train_on_csv, MODEL_PATH, init_db

app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app)

DB_PATH = 'energy.db'

# Ensure DB and model exist
init_db(DB_PATH)
ensure_model()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/history', methods=['GET'])
def history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, datetime, temperature, humidity, power FROM measurements ORDER BY datetime DESC LIMIT 200", conn)
    conn.close()
    return df.to_dict(orient='records')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # expected keys: datetime, temperature, humidity, hour (optional)
    pred = predict_single(data)
    # store prediction
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (datetime, temperature, humidity, hour, predicted_power) VALUES (?, ?, ?, ?, ?)",
        (data.get('datetime'), data.get('temperature'), data.get('humidity'), data.get('hour'), float(pred))
    )
    conn.commit()
    conn.close()
    return jsonify({'predicted_power': float(pred)})

@app.route('/api/upload', methods=['POST'])
def upload():
    # upload CSV to train model. CSV should have columns: datetime, temperature, humidity, power
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'no file provided'}), 400
    filepath = Path('data') / file.filename
    os.makedirs('data', exist_ok=True)
    file.save(str(filepath))
    # train model
    mse = train_on_csv(str(filepath))
    return jsonify({'status': 'trained', 'mse': mse})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
