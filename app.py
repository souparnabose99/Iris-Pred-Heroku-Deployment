from flask import Flask, request, jsonify

import numpy as numpy
from tensorflow.keras.models import load_model
import joblib

def model_prediction(model, scaler, sample_json):
    
    s_len = sample_json['sepal_length']
    s_width = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_width = sample_json['petal_width']
    
    flower = [[s_len, s_width, p_len, p_width]]
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    flower = scaler.transform(flower)
    
    class_index = model.predict_classes(flower)[0]
    
    return classes[class_index]

app = Flask(__name__)

@app.route("/")
def index():
	return '<h1>Flask app is running</h1>'


flower_model = load_model('iris_model_pred.h5')
flower_scaler = joblib.load('iris_scaler.pkl')


@app.route('/api/flower', method=['POST'])
def flower_prediction():

	content = request.json
	results = model_prediction(flower_model, flower_scaler, content)
	return jsonify(results)


if __name__ == '__main__':
	app.run()