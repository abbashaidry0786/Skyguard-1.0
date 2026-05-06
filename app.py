from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Paths
MODEL_PATH = 'india_air_quality_model_random_forest.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURE_INFO_PATH = 'feature_info.pkl'

# Initialize variables
model = None
scaler = None
feature_info = None
df = None

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Load scaler
try:
    scaler = joblib.load(SCALER_PATH)
    print("✓ Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading scaler: {e}")

# Load feature info
try:
    with open(FEATURE_INFO_PATH, 'rb') as f:
        feature_info = pickle.load(f)
    print("✓ Feature info loaded successfully!")
except Exception as e:
    print(f"Error loading feature info: {e}")
    feature_info = {
        'feature_columns': ['so2', 'no2', 'rspm', 'spm', 'year', 'month', 'state_encoded', 'type_encoded'],
        'target_column': 'pm2_5'
    }
    print("✓ Using default feature info")

# Load dataset
url = "https://drive.google.com/uc?id=1aqu7GIBTi-eHhMG2BUmteZtTNLfXfnDB"

try:
    df = pd.read_csv(url, low_memory=False, encoding='latin1', on_bad_lines='skip')

    for col in ['so2', 'no2', 'rspm', 'spm', 'pm2_5']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    print("✓ Data loaded successfully!")
    print(f"Dataset shape: {df.shape}")

except Exception as e:
    print(f"Error loading data: {e}")
    df = None


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/visualization')
def visualization():
    return render_template('visualization.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        data = request.get_json()

        so2 = float(data.get('so2', 0))
        no2 = float(data.get('no2', 0))
        rspm = float(data.get('rspm', 0))
        spm = float(data.get('spm', 0))
        year = int(data.get('year', datetime.now().year))
        month = int(data.get('month', datetime.now().month))

        input_data = {
            'so2': [so2],
            'no2': [no2],
            'rspm': [rspm],
            'spm': [spm],
            'year': [year],
            'month': [month]
        }

        if feature_info and 'feature_columns' in feature_info:
            if 'state_encoded' in feature_info['feature_columns']:
                input_data['state_encoded'] = [0]
            if 'type_encoded' in feature_info['feature_columns']:
                input_data['type_encoded'] = [0]

        input_df = pd.DataFrame(input_data)

        if feature_info and 'feature_columns' in feature_info:
            cols = [c for c in feature_info['feature_columns'] if c in input_df.columns]
            input_df = input_df[cols]

        prediction = model.predict(input_df)[0]

        if np.isnan(prediction) or np.isinf(prediction):
            prediction = 50.0

        category, color, impact = calculate_aqi_category(prediction)

        return jsonify({
            'success': True,
            'pm2_5_prediction': round(float(prediction), 2),
            'aqi_category': category,
            'aqi_color': color,
            'health_impact': impact
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


def calculate_aqi_category(pm25):
    try:
        pm25 = float(pm25)

        if pm25 <= 30:
            return "Good", "#00e400", "Low risk"
        elif pm25 <= 60:
            return "Satisfactory", "#ffff00", "Minor risk"
        elif pm25 <= 90:
            return "Moderate", "#ff7e00", "Sensitive groups affected"
        elif pm25 <= 120:
            return "Poor", "#ff0000", "Health effects possible"
        elif pm25 <= 250:
            return "Very Poor", "#8f3f97", "Serious health effects"
        else:
            return "Severe", "#7e0023", "Emergency conditions"

    except:
        return "Unknown", "#999999", "Error"


@app.route('/api/visualization-data')
def get_visualization_data():
    if df is None:
        return jsonify({'success': True, 'message': 'No data available'})

    return jsonify({
        'success': True,
        'rows': len(df)
    })


@app.route('/api/model-info')
def get_model_info():
    return jsonify({
        'success': True,
        'model': 'Random Forest',
        'features': feature_info['feature_columns'] if feature_info else []
    })


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)
