# AI-based Power Consumption Prediction System

## What is included
- backend: app.py (Flask) — serves API endpoints and static frontend
- model.py — training and prediction utilities (RandomForest)
- train_model.py — helper to train on a CSV
- data/sample_power_data.csv — small sample dataset
- static/ — frontend (index.html, script.js)
- requirements.txt

## Quick start (local)
1. Create a python virtualenv and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```
2. Option A: Train model with sample data:
   ```
   python train_model.py data/sample_power_data.csv
   ```
   This will create `model.joblib`.
3. Run the Flask app:
   ```
   python app.py
   ```
   Open http://localhost:5000

4. Use the frontend to make predictions or upload your own CSV via /api/upload (in a REST client).

## Notes
- CSV format for training: datetime,temperature,humidity,power
- The backend stores measurements and predictions in `energy.db` (SQLite).
- For production, consider using a proper web server (gunicorn), secure the API, and use a production DB.
