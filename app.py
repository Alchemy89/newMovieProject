from flask import Flask, render_template, url_for, request, jsonify
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
    
    nonLinGraph = NonLinear.calculate()
    correlate = Linear.correlate().to_frame().to_html()
    
    indep = request.form.get('indep')
    if indep:
        LinGraph = Linear.plotChart(str(indep))
        return LinGraph
    else:
        return render_template('project_data.html', nonLinGraph=nonLinGraph, corr=correlate)

    
    
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
        budget = request.form.get('budget')
        genre = request.form.get('genre')
        popularity = request.form.get('popular')
        vote_cnt = request.form.get('vote')

        print(budget)
        print(popularity)
        print(vote_cnt)
        dataTrain = pd.read_csv('./tmdb_5000_train.csv')
        dataTest = pd.read_csv('./tmdb_5000_test.csv')
        x_train = dataTrain[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
        y_train = dataTrain['revenue']
        ols = LinearRegression()
        model = ols.fit(x_train, y_train)

        input = {'budget': float(budget), 'popularity': float(popularity), 'vote_count': float(vote_cnt)}
        X = pd.DataFrame.from_dict(input,orient='index')
        X = X.values.reshape(-1, 3)

        answer = model.predict(X)
        return jsonify({'answer' : answer[0]})
    #return '''<h1>This is budget: {}</h1>
    #              <h1>This is genre: {}</h1>
    #              <h1>This is popularity: {}</h1>
    #              <h1>This is vote_cnt: {}</h1>
    #              <h1>Prediction is: {}</h1>
    #              '''.format(budget, genre, popularity, vote_cnt, (float(answer)))



if __name__ == '__main__':
    app.run(debug=True)
