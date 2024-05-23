from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

# ----------------- Setup -----------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    data = req['features']
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

# ----------------- Run -----------------

if __name__ == '__main__':
    app.run(debug=True)