"""
Wine Cultivar Origin Prediction System - Flask Web Application
Author: [Your Name]
Matric No: [Your Matric Number]
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and selected features
MODEL_PATH = os.path.join('model', 'wine_cultivar_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
FEATURES_PATH = os.path.join('model', 'selected_features.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_features = joblib.load(FEATURES_PATH)
    print("Model loaded successfully!")
    print(f"Selected features: {selected_features}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    selected_features = None

# Cultivar names mapping
CULTIVAR_NAMES = {
    0: "Cultivar 0 (Class 1)",
    1: "Cultivar 1 (Class 2)",
    2: "Cultivar 2 (Class 3)"
}

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded properly. Please check the model files.'
            }), 500
        
        # Get input data from the form
        features_data = []
        input_dict = {}
        
        for feature in selected_features:
            value = float(request.form.get(feature, 0))
            features_data.append(value)
            input_dict[feature] = value
        
        # Convert to numpy array and reshape
        features_array = np.array(features_data).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get cultivar name
        cultivar_name = CULTIVAR_NAMES.get(prediction, f"Cultivar {prediction}")
        
        # Prepare probability data
        probabilities = {
            CULTIVAR_NAMES[i]: f"{prob*100:.2f}%" 
            for i, prob in enumerate(prediction_proba)
        }
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'cultivar_name': cultivar_name,
            'confidence': f"{max(prediction_proba)*100:.2f}%",
            'probabilities': probabilities,
            'input_features': input_dict
        }
        
        return jsonify(result)
    
    except ValueError as ve:
        return jsonify({
            'error': f'Invalid input: {str(ve)}. Please enter valid numbers.'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/info')
def model_info():
    """Return model information"""
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'features': selected_features,
        'cultivars': list(CULTIVAR_NAMES.values())
    })

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)