import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sqlite3

MODEL_PATH = 'model.joblib'

def init_db(db_path='energy.db'):
    # create sqlite db with sample tables if not exist
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS measurements
                (id INTEGER PRIMARY KEY AUTOINCREMENT, datetime TEXT, temperature REAL, humidity REAL, power REAL)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions
                (id INTEGER PRIMARY KEY AUTOINCREMENT, datetime TEXT, temperature REAL, humidity REAL, hour INTEGER, predicted_power REAL)''')
    conn.commit()
    conn.close()

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        # try to train on sample data if exists
        sample = 'data/sample_power_data.csv'
        if os.path.exists(sample):
            train_on_csv(sample)
        else:
            # create a trivial model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            # train on trivial synthetic data
            X = np.array([[0,20,50],[1,21,48],[2,19,55],[3,18,60]])
            y = np.array([0.5,0.6,0.4,0.3])
            model.fit(X,y)
            joblib.dump({'model':model, 'scaler':None}, MODEL_PATH)

def features_from_row(row):
    # expects dict-like with keys temperature, humidity, hour (if hour missing derive from datetime)
    temp = float(row.get('temperature', 0))
    hum = float(row.get('humidity', 0))
    hour = row.get('hour')
    if hour is None:
        # try parse hour from datetime
        dt = row.get('datetime')
        try:
            import pandas as pd
            hour = pd.to_datetime(dt).hour
        except:
            hour = 0
    return [hour, temp, hum]

def predict_single(row):
    data = features_from_row(row)
    saved = joblib.load(MODEL_PATH)
    model = saved['model']
    scaler = saved.get('scaler')
    X = np.array([data])
    if scaler is not None:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    return float(pred)

def train_on_csv(csv_path):
    df = pd.read_csv(csv_path)
    # require columns datetime, temperature, humidity, power
    df = df.dropna(subset=['temperature','humidity','power'])
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    X = df[['hour','temperature','humidity']].values
    y = df['power'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    joblib.dump({'model':model, 'scaler':scaler}, MODEL_PATH)
    return float(mse)
