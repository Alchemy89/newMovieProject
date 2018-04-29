from flask import Flask, render_template, url_for, request, flash, redirect, url_for, session, logging
from flask_bootstrap import Bootstrap
from flask import request
from flaskext.mysql import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
app=Flask(__name__)

# Config MySQL
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
#initialize mysql

mysql = MySQL()
mysql.init_app(app)

Bootstrap(app)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/project_data')
def projectdata():
    return render_template('project_data.html')

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

class RegisterForm(Form):
    name = StringField('Name', [validators.length(min=1, max=50)])
    username = StringField('Username', [validators.length(min=4, max= 20)])
    email = StringField('Email', [validators.length(min=5, max = 50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message="Passwords do not match"),
        ])
    confirm = PasswordField('Confirm Password')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        # //encrypt password
        password = sha256_crypt.encrypt(str(form.password.data))
        #create cursor
        cur = mysql.get_db().cursor()
        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)" , (name, email, username, password))
        mysql.connection.commit()
        cur.close()
        flash('Registration successful and can login!', 'success')
        redirect(url_for('index'))
    return render_template('register.html', form=form)




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
    app.secret_key='secret123'
