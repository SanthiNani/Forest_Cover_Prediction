from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('forest_cover_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {
        "Elevation": float(request.form['Elevation']),
        "Aspect": float(request.form['Aspect']),
        "Slope": float(request.form['Slope']),
        "Horizontal_Distance_To_Hydrology": float(request.form['Horizontal_Distance_To_Hydrology']),
        "Vertical_Distance_To_Hydrology": float(request.form['Vertical_Distance_To_Hydrology']),
        "Horizontal_Distance_To_Roadways": float(request.form['Horizontal_Distance_To_Roadways']),
        "Hillshade_9am": float(request.form['Hillshade_9am']),
        "Hillshade_Noon": float(request.form['Hillshade_Noon']),
        "Hillshade_3pm": float(request.form['Hillshade_3pm']),
        "Horizontal_Distance_To_Fire_Points": float(request.form['Horizontal_Distance_To_Fire_Points']),
        "Wilderness_Area1": 0,
        "Wilderness_Area2": 0,
        "Wilderness_Area3": 0,
        "Wilderness_Area4": 0,
        **{f"Soil_Type{i+1}": 0 for i in range(40)}
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale numerical features
    numerical_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make prediction
    prediction = model.predict(input_df)
    cover_types = {
        1: 'Spruce/Fir',
        2: 'Lodgepole Pine',
        3: 'Ponderosa Pine',
        4: 'Cottonwood/Willow',
        5: 'Aspen',
        6: 'Douglas-fir',
        7: 'Krummholz'
    }
    predicted_cover = cover_types[prediction[0]]

    # Render the result in the HTML template
    return render_template('index.html', prediction=predicted_cover)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)