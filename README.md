---

# Forest Cover Prediction using RandomForestClassifier and Flask

This project predicts forest cover types based on geographical and environmental features using a **RandomForestClassifier**. The model is deployed as a **Flask web application**, allowing users to input data through a web interface and get predictions.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Folder Structure](#folder-structure)
6. [Technologies Used](#technologies-used)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Project Overview**
The goal of this project is to predict forest cover types based on features such as elevation, slope, distance to hydrology, and more. The model is trained using a **RandomForestClassifier** and deployed using **Flask** to provide a user-friendly web interface for predictions.

---

## **Features**
- **Machine Learning Model**: RandomForestClassifier for accurate predictions.
- **Web Interface**: A simple HTML form for users to input data.
- **Scalability**: The Flask app can be easily deployed to production environments.
- **User-Friendly**: Predictions are displayed directly on the web page.

---

## **Installation**
Follow these steps to set up the project on your local machine.

### **Prerequisites**
- Python 3.x
- pip (Python package installer)

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/forest-cover-prediction.git
   cd forest-cover-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (`forest_cover_prediction.csv`) and place it in the project folder.

4. Train the Model and save the models:
   ```bash
   python save_models.py
   ```
- It will save the models with **.pkl** format.
- Models will save as **forest_cover_model.pkl** and **scaler.pkl**.
  
6. Start the Flask app:
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000` to access the web interface.

---

## **Usage**
1. **Access the Web Interface**:
   - Open your browser and navigate to `http://127.0.0.1:5000`.

2. **Input Data**:
   - Fill out the form with the required features (e.g., elevation, slope, distance to hydrology, etc.).

3. **Get Predictions**:
   - Click the "Predict" button to see the predicted forest cover type.

---

## **Folder Structure**
```
forest-cover-prediction/
│
├── app.py                  # Flask application
├── forest_cover_model.pkl  # Trained RandomForestClassifier model
├── scaler.pkl              # Scaler used for preprocessing
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── templates/
    └── index.html          # HTML form for user input
```

---

## **Technologies Used**
- **Python**: Programming language used for the project.
- **Flask**: Web framework for deploying the model.
- **Scikit-learn**: Machine learning library for training the RandomForestClassifier.
- **Pandas**: Data manipulation and analysis.
- **HTML/CSS**: For creating the web interface.

---

## **Outputs For this Project** ##
![Screenshot 2025-03-04 200354](https://github.com/user-attachments/assets/ee0b6d05-d956-4747-8705-7fb2b93fc527)
![Screenshot 2025-03-04 201047](https://github.com/user-attachments/assets/c756cbbb-24b3-4c6c-a17e-1ace1028f12d)
![Screenshot 2025-03-04 201106](https://github.com/user-attachments/assets/457970d5-d701-49ad-838d-9a8742b2b084)

## **Contributing**
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your branch and submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- Dataset: [Forest Cover Type Dataset](https://archive.ics.uci.edu/ml/datasets/Covertype)
- Flask Documentation: [Flask Official Docs](https://flask.palletsprojects.com/)
- Scikit-learn Documentation: [Scikit-learn Official Docs](https://scikit-learn.org/stable/)

---

## **Contact**
For questions or feedback, feel free to reach out:
- **Name**: Peddapati Santhi raju  
- **Email**: santhinani364@gmail.com
- **GitHub**:SanthiNani (https://github.com/SanthiNani)

---

