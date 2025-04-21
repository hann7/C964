from flask import Flask, request, jsonify, send_from_directory
from app.server.model.predict import predict_severity
from PIL import Image
import numpy as np

app = Flask(__name__, static_folder='../client', static_url_path='')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream)

    result = predict_severity(image)
    result = int(result) if isinstance(result, np.int64) else result
    
    return jsonify({"severity": result})

if __name__ == '__main__':
    app.run(debug=True)
