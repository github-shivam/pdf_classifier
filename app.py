from flask import Flask, request, jsonify
import p_model
import joblib

vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')
model = joblib.load('best_classifier.sav')

app = Flask(__name__)

@app.route('/predict', methods = ['Post'])

def predict():
    try:


        data = request.json
        pdf_url = data.get('pdf_url')

        if not pdf_url:
            return jsonify({'error': "No PDF URL Provided"}), 400
    
        result = p_model.prediction(pdf_url)

        predicted_class = result['predicted_class']
        predicted_class_probability = float(result['predicted_class_probability'])
        class_probabilities = [float(prob) for prob in result['class_probabilities']]


        return jsonify({
            'predicted_class': predicted_class,
            'predicted_class_probability': predicted_class_probability,
            'class_probabilities': class_probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
