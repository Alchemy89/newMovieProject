

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix as scm
from numpy import *
from scipy import stats
from numpy import set_printoptions
from io import BytesIO
import base64


def correlate():
    
    data = pd.read_csv('./tmdb_5000.csv')

    
    #Select the Columns that ONLY Use NUMBERS
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdata = data.select_dtypes(include=numdtypes)
    
    #Remove the id column. It's useless for us
    #numdata = numdata.drop(['id'], axis=1)
    
    #print all the columns that have numerical data
    
    #Command Prompt asking for input
    #columnDep = input("Please type Dependent Variable:")
    
    #print the p-value correlation
    corr = numdata[numdata.columns[0:]].corr()['revenue']
    corr = corr.drop(['revenue'])
    
    #print(corr.loc[corr.gt(0.5)])
    
    return corr;
    
    
#UNCOMMENT TO RUN


def modelSummary(indep):
    

    data = pd.read_csv('./tmdb_5000.csv')

    
    #Select the Columns that ONLY Use NUMBERS
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdata = data.select_dtypes(include=numdtypes)
    
    #Remove the id column. It's useless for us
    #numdata = numdata.drop(['id'], axis=1)
        
    
    #Command Prompt asking for input
    columnInd = str(indep)
    
    x = numdata[columnInd]
    y = numdata['revenue']
    
    model = sm.OLS(y, x).fit()
    return model.summary()

#modelSummary('budget')


def plotChart(indep): 
    data = pd.read_csv('./tmdb_5000.csv')

    
    #Select the Columns that ONLY Use NUMBERS
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdata = data.select_dtypes(include=numdtypes)
    
    #Remove the id column. It's useless for us
    #numdata = numdata.drop(['id'], axis=1)
    columnInd = str(indep)
    
    
    #Run Linear Analysis
    plt.figure()
    x = numdata[columnInd]
    y = numdata['revenue']
    numdata.plot.scatter(columnInd, 'revenue')
    
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
        
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


def multiRegChart():
    dataTrain = pd.read_csv('./tmdb_5000_train.csv')
    dataTest = pd.read_csv('./tmdb_5000_test.csv')    
    
    x_train = dataTrain[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
    y_train = dataTrain['revenue']
    
    x_test = dataTest[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
    y_test = dataTest['revenue']

    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    
    dataTrain = pd.read_csv('./tmdb_5000.csv', usecols=['budget','popularity','runtime','vote_average','vote_count','IMDB','rotten','metaC','revenue'])
    dataTest = pd.read_csv('./tmdb_5000_test.csv', usecols=['budget','popularity','runtime','vote_average','vote_count','IMDB','rotten','metaC','revenue'])    
    
    
    y_predicted = model.predict(x_train)
    
    plt.figure()
    plt.scatter(y_train, y_predicted)
    plt.plot(y_train,y_predicted,'o')
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
        
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


def multiRegPValue():
    
    #IDEAL IS FOR THE INDEPENDENT VARIABLE TO BE CORRELATED WITH THE DEPENDENT VARIABLE BUT NOT 
    #WITH EACH OTHER
    #Select the Columns that ONLY Use NUMBERS
    dataTrain = pd.read_csv('./tmdb_5000_train.csv')
    dataTest = pd.read_csv('./tmdb_5000_test.csv')    
    
    x_train = dataTrain[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
    y_train = dataTrain['revenue']
    
    x_test = dataTest[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
    y_test = dataTest['revenue']

    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    
    dataTrain = pd.read_csv('./tmdb_5000.csv', usecols=['budget','popularity','runtime','vote_average','vote_count','IMDB','rotten','metaC','revenue'])
    dataTest = pd.read_csv('./tmdb_5000_test.csv', usecols=['budget','popularity','runtime','vote_average','vote_count','IMDB','rotten','metaC','revenue'])    
    

    x_train = dataTrain[['rotten', 'IMDB', 'vote_average']].values.reshape(-1,3)    
    y_train = dataTrain['revenue']
    
    x_test = dataTest[['rotten', 'IMDB', 'vote_average']].values.reshape(-1,3)
    y_test = dataTest['revenue']

    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    params = np.append(model.intercept_,model.coef_)
    
    predictions = model.predict(x_train)
    #print(predictions)
    
    newX = pd.DataFrame({"Constant":np.ones(len(x_test))}).join(pd.DataFrame(x_test))
    MSE = (sum((y_train-predictions)**2))/(len(newX)-len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)    
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()  
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
    
    return myDF3

    
    
    #print(model.predict(x_test)[0:10])
#UNCOMMENT TO RUN
#multipleregress()

