from flask import Flask, request, jsonify, json
import numpy as np
import joblib  # Use joblib for loading the model
from flask_cors import CORS

# Load the trained DecisionTreeClassifier model using joblib
loaded_model = joblib.load('models.joblib')

# Create a Flask app
app = Flask(__name__)

# CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

# Define a Flask route to expose the ML model as an endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json().get('input_data') 
    print(input_data)
    input_data_df = np.array([input_data], dtype=np.float32)

    
    # Make predictions using the ML model
    classifier_duofos = loaded_model[0]
    classifier_urea = loaded_model[1]
    classifier_muriate = loaded_model[2]
    
    duofos_prediction = classifier_duofos.predict(input_data_df)
    urea_prediction = classifier_urea.predict(input_data_df)
    muriate_prediction = classifier_muriate.predict(input_data_df)
     
    # Return the predictions as a JSON response
    return json.dumps({
    'Duofos(0-20-0)': duofos_prediction[0],
    'Urea(46-0-0)': urea_prediction[0],
    'Muriate of Potash(0-0-60)': muriate_prediction[0]
    })

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
