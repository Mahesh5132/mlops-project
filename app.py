import pickle
from flask import Flask, request, jsonify, render_template

# Load the ML model
model = pickle.load(open("model.pkl", "rb"))

# Flask App
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return render_template('index.html')

@flask_app.route('/predict', methods=['POST'])
def predict_flask():
    try:
        data = request.form.get("features")  # Get input as a string
        features = list(map(float, data.split(",")))  # Convert to list of floats
        prediction = model.predict([features])
        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    flask_app.run(host="127.0.0.1", port=5000)