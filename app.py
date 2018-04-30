from flask import Flask, render_template, url_for, request, flash, redirect, url_for, session, logging, jsonify
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from sklearn.linear_model import LinearRegression
from passlib.hash import sha256_crypt
from flask_bootstrap import Bootstrap
from functools import wraps
from flask import request
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import scipy
import NonLinear
import Linear

app=Flask(__name__)

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

#initialize mysql
mysql = MySQL()
mysql.init_app(app)

Bootstrap(app)
@app.route('/')
def index():
    return render_template('index.html')

def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Please login', 'danger')
            return redirect(url_for('login'))
    return wrap


@app.route('/predict', methods=['POST', 'GET'])
@is_logged_in
def predict():
    return render_template('predict.html')

@app.route('/project_data', methods=['POST', 'GET'])
def projectdata():

    nonLinGraph = NonLinear.sendgraph()
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

class RegisterForm(Form):
    name = StringField('Name', [validators.length(min=1, max=50)])
    username = StringField('Username', [validators.length(min=4, max= 20)])
    email = StringField('Email', [validators.length(min=5, max = 50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message="Passwords do not match"),
        ])
    confirm = PasswordField('Confirm Password')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']
        # Create cursor
        cur = mysql.connection.cursor()
        # Get user by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username

                flash('You are now logged in', 'success')
                return redirect(url_for('predict'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')

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
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)" , (name, email, username, password))
        mysql.connection.commit()
        cur.close()
        flash('Registration successful and can login!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', form=form)

@app.route('/nonlinear', methods=['POST'])
def predict_with_nonlinear():
     if request.method == 'POST':
        budget = request.form.get('budget')
        genre = request.form.get('genre')
        popularity = request.form.get('popular')
        vote_cnt = request.form.get('vote')

        answer = NonLinear.predict(budget, popularity, vote_cnt)
        print(answer)
        print(type(answer))
        return jsonify({'nonlinAns' : round(answer,2)})

@app.route('/linear', methods=['POST'])
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
        return jsonify({'answer' : round(answer[0],2)})
    #return '''<h1>This is budget: {}</h1>
    #              <h1>This is genre: {}</h1>
    #              <h1>This is popularity: {}</h1>
    #              <h1>This is vote_cnt: {}</h1>
    #              <h1>Prediction is: {}</h1>
    #              '''.format(budget, genre, popularity, vote_cnt, (float(answer)))



if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
