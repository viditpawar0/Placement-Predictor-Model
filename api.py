from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('placement_predictor_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict placement status and probability based on input parameters.
    Expects JSON input with keys:
    - CGPA, Internships, Projects, Workshops_Certifications, AptitudeTestScore, SoftSkillsRating
    """
    data = request.json
    input_data = [[
        data['CGPA'],
        data['Internships'],
        data['Projects'],
        data['Workshops_Certifications'],
        data['AptitudeTestScore'],
        data['SoftSkillsRating']
    ]]
    predicted_class = model.predict(input_data)[0]
    predicted_probability = model.predict_proba(input_data)[0][1]
    return jsonify({
        'PredictedClass': int(predicted_class),
        'ProbabilityOfPlacement': round(predicted_probability, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
