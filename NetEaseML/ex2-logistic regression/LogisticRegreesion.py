#encoding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def get_training_data():

    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    #print (data.head())

    #print (data['Admitted'])
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    #plt.show()
    
    # add a ones column - this makes the matrix multiplication work out easier
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    # convert to numpy arrays and initalize the parameter array theta
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros((1,3))
    return X,y,theta,data

def cost(theta, X, y):

    theta = np.matrix(theta)
    first = np.multiply(-y , np.log(sigmoid(np.dot(X , theta.T))))
    second = np.multiply((1 - y), np.log(1 - sigmoid(np.dot(X , theta.T))))
    return np.sum(first - second) / (len(X))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(theta, X, y):
    
    m = X.shape[0]
    inner = np.dot(sigmoid(np.dot(theta ,X.T)) - y.T,X)  # (1,m) @ (m, n) -> (1, n)
    return inner / m

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def get_regularization_training_data():

    path  =  'ex2data2.txt'
    data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    #print (data2.head())

    positive = data2[data2['Accepted'].isin([1])]
    negative = data2[data2['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    #plt.show()

    degree = 5
    x1 = data2['Test 1']
    x2 = data2['Test 2']

    data2.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(0, i):
            data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

    data2.drop('Test 1', axis=1, inplace=True)
    data2.drop('Test 2', axis=1, inplace=True)

    #print (data2.head())

    # set X and y (remember from above that we moved the label to column 0)
    cols = data2.shape[1]
    X2 = data2.iloc[:,1:cols]
    y2 = data2.iloc[:,0:1]

    # convert to numpy arrays and initalize the parameter array theta
    X2 = np.array(X2.values)
    y2 = np.array(y2.values)
    theta2 = np.zeros(11)

    return X2,y2,theta2

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    cost_non_reg=cost(theta, X, y)
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:], 2))
    return cost_non_reg + reg

def gradientReg(theta, X, y, learningRate):
    lamda = 2
    #  '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = np.multiply((learningRate * lamda/ len(X)) , theta_j1_to_n)

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term
 
######################################

def test_sigmoid():
    nums = np.arange(-10, 10, step=1)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(nums, sigmoid(nums), 'r')
    plt.show()

def test_predict():
    X,y,theta,data =get_training_data()

    print (cost(theta, X, y))
    print (gradient(theta, X, y)) 
    result = opt.fmin_tnc(func=cost, x0=theta , fprime=gradient, args=(X, y))
    print (result)

    coef = -(result[0] / result[0][2])  # find the equation
    print(coef)

    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, X)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print ('accuracy = {0}%'.format(accuracy))

def test_regularization():
    learningRate = 1
    
    X2, y2 ,theta2 = get_regularization_training_data()
    print (costReg(theta2, X2, y2, learningRate))
    print (gradientReg(theta2, X2, y2, learningRate))
    result2 = opt.fmin_tnc(func=costReg, x0=np.zeros(11), fprime=gradientReg, args=(X2, y2, learningRate))
    print (result2)

    theta_min = np.matrix(result2[0])
    predictions = predict(theta_min, X2)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print ('accuracy = {0}%'.format(accuracy))

def test_regularization_with_sk_learn():

    from sklearn import linear_model#调用sklearn的线性回归包
    X2, y2 ,theta2 = get_regularization_training_data()

    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(X2, y2.ravel())
    print (model.score(X2, y2))

# test_sigmoid()
# test_predict()
#####regularization########
test_regularization()
test_regularization_with_sk_learn()

