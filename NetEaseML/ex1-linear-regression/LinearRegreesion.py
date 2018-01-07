#encoding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def get_training_data():

    path =  'ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    #print (data.head())
    #print (data.describe())

    # data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
    # plt.show()
    
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
    y = data.iloc[:,cols-1:cols]#X是所有行，最后一列
    #print (X.head())#head()是观察前5行

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    
    #print  (X.shape, theta.shape, y.shape)
    return X,y,data

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradient(theta, X, y):
    m = X.shape[0]

    inner = np.dot(np.dot(theta ,X.T) - y.T,X)  # (1,m) @ (m, n) -> (1, n)
    return inner / m

def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters)
    m = X.shape[0]
    for i in range(iters):
        cost[i] = computeCost(X, y, theta)
        theta  =  theta - alpha * gradient(theta, X, y)
        
    return theta, cost

def get_training_data_multi():
    path =  'ex1data2.txt'
    data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    data2.head()   
    data2 = (data2 - data2.mean()) / data2.std()
    data2.head()
    # add ones column
    data2.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data2.shape[1]
    X2 = data2.iloc[:,0:cols-1]
    y2 = data2.iloc[:,cols-1:cols]

    # convert to matrices and initialize theta
    X2 = np.matrix(X2.values)
    y2 = np.matrix(y2.values)
    return X2,y2,data2

###########################
def test_computeCost():
    X,y,data = get_training_data()
    theta = np.zeros((1,2))
    print  (computeCost(X, y, theta))
    
def test_gradientDescent():
    alpha = 0.01
    iters=1000 
    X,y,data= get_training_data()
    theta = np.matrix(np.array([0,0]))
    
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    X.shape, theta.shape, y.shape
    print (g)

    ####plt result
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

    print (computeCost(X, y, g))
    ####plt cost
    fig, ax = plt.subplots(figsize=(12,8))
    #ax.axis([-1,iters,1, 30])  
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()

def test_gradientDescent_multi():
    alpha = 0.01
    iters=1000 
    X2,y2,data2= get_training_data_multi()

    theta2 = np.zeros((1,3))

    # perform linear regression on the data set
    g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

    # get the cost (error) of the model
    print (computeCost(X2, y2, g2))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(iters), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
    
def test_with_sklearn():
    X,y,data= get_training_data()
    model = linear_model.LinearRegression()
    model.fit(X, y)   
    x = np.array(X[:, 1].A1)
    f = model.predict(X).flatten()

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()
    print ((model.intercept_,model.coef_))

def test_learning_rate():

    X,y,data= get_training_data()

    base = np.logspace(-3, -5, num=4)
    candidate = np.sort(np.concatenate((base, base*3)))
    print(candidate)
    epoch=50

    fig, ax = plt.subplots(figsize=(16, 9))

    for alpha in candidate:
        theta = np.zeros((1,2))
        _, cost_data = gradientDescent( X, y,theta, alpha,epoch)
        ax.plot(np.arange(epoch), cost_data, label=alpha)

    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('cost', fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title('learning rate', fontsize=18)
    plt.show()

#test_computeCost()    
#test_gradientDescent()
#test_gradientDescent_multi()
#test_with_sklearn()
test_learning_rate()

