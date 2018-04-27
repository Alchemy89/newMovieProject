from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask import request
import pandas as pd
import numpy as np
import scipy
import NonLinear
import Linear
from sklearn.linear_model import LinearRegression
app=Flask(__name__)

Bootstrap(app)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/project_data', methods=['POST', 'GET'])
def projectdata():
    if request.method == 'POST':    
        indep = request.form('indep')
    
        if indep:
            Linear.simpleregress('indep')
    
    nonLinGraph = NonLinear.calculate()
    return render_template('project_data.html', nonLinGraph=nonLinGraph)
    
@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/past_movies')
def past():
    return render_template('past_movies.html')

@app.route('/nonlinear', methods=['POST'])
def predict_with_nonlinear():
    # add your alogrithm here
    print(null)

@app.route('/predict', methods=['POST'])
def predict_with_linear():
    if request.method == 'POST':
        budget = request.form.get('namequery')
        genre = request.form.get('genrequery')
        popularity = request.form.get('popquery')
        vote_cnt = request.form.get('votequery')

        dataTrain = pd.read_csv('./tmdb_5000_train.csv')
        dataTest = pd.read_csv('./tmdb_5000_test.csv')
        x_train = dataTrain[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
        y_train = dataTrain['revenue']
        ols = LinearRegression()
        model = ols.fit(x_train, y_train)

        input = {'budget': float(budget), 'popularity': float(popularity), 'vote_count': float(vote_cnt)}
        X = pd.DataFrame.from_dict(input,orient='index')
        print(X)
        X = X.values.reshape(-1, 3)

        answer = model.predict(X)


    return '''<h1>This is budget: {}</h1>
                  <h1>This is genre: {}</h1>
                  <h1>This is popularity: {}</h1>
                  <h1>This is vote_cnt: {}</h1>
                  <h1>Prediction is: {}</h1>
                  '''.format(budget, genre, popularity, vote_cnt, (float(answer)))



if __name__ == '__main__':
    app.run(debug=True)
