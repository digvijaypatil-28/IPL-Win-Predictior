from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load your trained model (replace with your model's path)
model = pickle.load(open('pipe2.pkl', 'rb'))

# Create a column transformer for encoding categorical columns
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['batting_team', 'bowling_team', 'venue'])
    ], remainder='passthrough'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    venue = request.form['venue']
    target = int(request.form['target'])
    runs_scored = int(request.form['runs_scored'])
    overs_completed = float(request.form['overs_completed'])
    wickets_down = int(request.form['wickets_down'])
    
    runs_left = target - runs_scored
    balls_bowled = overs_completed * 6
    balls_left = 120 - balls_bowled
    wickets_remaining = 10 - wickets_down
    crr = runs_scored / overs_completed if overs_completed > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    # Prepare the input data for prediction
    # input_data = np.array([[batting_team, bowling_team, venue, target, runs_scored, overs_completed, wickets_down]])
    input_data = pd.DataFrame([{
    'batting_team': batting_team,
    'bowling_team': bowling_team,
    'city': venue,
    'runs_left': runs_left,
    'balls_left': balls_left,
    'wickets_remaining': wickets_remaining,
    'total_runs_x': target,
    'crr': crr,
    'rrr': rrr,
}])

    # Transform the input data using the column transformer
    # transformed_data = column_transformer.fit_transform(input_data)

    # Predict the winner using the loaded model
    # prediction = model.predict(transformed_data)[0]
    prediction = model.predict(input_data)[0]
    print("Prediction:",prediction)
    print(type(prediction))

    winner = batting_team if prediction == 1 else bowling_team


    # Return the result to the HTML page
    return render_template('index.html', prediction_result=winner)
@app.route('/result')
def result():
    return render_template('result.html', prediction="Dummy Prediction")

if __name__ == '__main__':
    app.run(debug=True)
