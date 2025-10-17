import pickle
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('MLR_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        rnd = float(request.form['rnd'])
        admin = float(request.form['admin'])
        marketing = float(request.form['marketing'])
        state = int(request.form['state'])  # 0: NY, 1: CA, 2: FL

        # Input to model: [R&D, Admin, Marketing, State]
        features = np.array([[rnd, admin, marketing, state]])

        # Prediction
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

        return render_template('index.html', prediction=prediction)
    
    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)
