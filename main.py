from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import logging
import os
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load the trained model
try:
    model = joblib.load('sentiment_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def clean_text(text):
    """Clean and preprocess text input"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def validate_input(text):
    """Validate text input"""
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) > 5000:
        return False, "Text too long (max 5000 characters)"
    
    if len(text.strip()) < 3:
        return False, "Text too short (min 3 characters)"
    
    return True, ""

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Predict sentiment for given text"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'success': False
            }), 500
        
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'success': False
            }), 400
        
        text = data['text']
        
        # Validate input
        is_valid, error_msg = validate_input(text)
        if not is_valid:
            return jsonify({
                'error': error_msg,
                'success': False
            }), 400
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Make prediction
        prediction = model.predict([cleaned_text])[0]
        confidence = model.predict_proba([cleaned_text])[0]
        
        # Prepare response
        sentiment = "positive" if prediction == 1 else "negative"
        confidence_score = float(max(confidence))
        
        logger.info(f"Prediction made: {sentiment} (confidence: {confidence_score:.3f})")
        
        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'confidence': confidence_score,
            'text': cleaned_text,
            'emoji': "😊" if sentiment == "positive" else "😢"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple texts"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'success': False
            }), 500
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'No texts provided',
                'success': False
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'error': 'Texts must be a list',
                'success': False
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'error': 'Too many texts (max 100)',
                'success': False
            }), 400
        
        results = []
        for i, text in enumerate(texts):
            is_valid, error_msg = validate_input(text)
            if not is_valid:
                results.append({
                    'index': i,
                    'error': error_msg,
                    'success': False
                })
                continue
            
            cleaned_text = clean_text(text)
            prediction = model.predict([cleaned_text])[0]
            confidence = model.predict_proba([cleaned_text])[0]
            
            sentiment = "positive" if prediction == 1 else "negative"
            confidence_score = float(max(confidence))
            
            results.append({
                'index': i,
                'success': True,
                'sentiment': sentiment,
                'confidence': confidence_score,
                'text': cleaned_text,
                'emoji': "😊" if sentiment == "positive" else "😢"
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)