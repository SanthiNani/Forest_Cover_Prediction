import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
from flask import Flask, request, jsonify

# Step 1: Load the dataset
data = pd.read_csv('forest_cover_prediction.csv')

# Drop the 'Id' column if it exists
if 'Id' in data.columns:
    data = data.drop('Id', axis=1)

# Step 2: Separate features and target
X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# Step 3: Scale numerical features
numerical_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                  'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                  'Horizontal_Distance_To_Fire_Points']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Step 6: Best model
best_model = grid_search.best_estimator_

# Step 7: Evaluate the model
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the model and scaler
joblib.dump(best_model, 'forest_cover_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use

# Step 9: Load the model and scaler
model = joblib.load('forest_cover_model.pkl')
scaler = joblib.load('scaler.pkl')

# Step 10: Initialize Flask app
app = Flask(__name__)

# Step 11: Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    input_data = request.get_json()
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure the columns are in the same order as during training
    input_df = input_df[X.columns]
    
    # Scale numerical features
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
    
    # Return the prediction as JSON
    return jsonify({'predicted_forest_cover_type': predicted_cover})

# Step 12: Run the Flask app
if __name__ == '__main__':
    # Train and save the model
    print("Training the model...")
    # (The training steps are already included above)
    
    # Start the Flask app
    print("Starting the Flask app...")
    app.run(debug=True)