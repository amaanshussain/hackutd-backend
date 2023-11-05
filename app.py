from flask import Flask, request
from flask_cors import CORS
import os
from EOGDetection import EOGDetection


app = Flask(__name__)
CORS(app)


EOG = EOGDetection()


@app.route('/', methods=["GET"])
def home():
    return 'Hello, World!'


@app.route("/load", methods=["POST"])
def load():
    
    sensor = request.files["sensor"]
    weather = request.files.get("weather")
    leaks = request.files.get("leaks")

    sensor.save(os.path.join('./sensor.csv'))
    weather.save(os.path.join('./weather.csv'))
    leaks.save(os.path.join('./leaks.csv'))

    EOG.cleanup_data('./weather.csv', './leaks.csv', './sensor.csv')

    data = EOG.sensor_classification('./sensor_trained.csv', './weather_cleaned.csv')


    return {"response": "Success", "data": data.to_dict()}


@app.route("/predict_leaks", methods=["GET"])
def predict_leaks():

    data = EOG.predict_leaks('./leaks_cleaned.csv')

    return data

@app.route("/leaks_status", methods=["GET"])
def leaks_status():

    data = EOG.leak_status('./leaks_cleaned.csv')

    return data

if __name__ == "__main__":
    port = 5000
    app.run(debug=True, host='0.0.0.0', port=port)