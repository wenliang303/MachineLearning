#encoding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
import math 
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

def get_training_data(path):

    raw_data = loadmat(path)
    data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
    data['y'] = raw_data['y']
    return data

def visual_data(path):
    data = get_training_data(path)
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
    ax.legend()
    plt.show()

def linear_svc(c=1):
    data = get_training_data('data/ex6data1.mat')
    svc = svm.LinearSVC(C=c, loss='hinge', max_iter=1000)
    print (svc)

    svc.fit(data[['X1', 'X2']], data['y'])
    print (svc.score(data[['X1', 'X2']], data['y']))

    x_min,x_max=get_min_max(data['X1'],data['X2'])
    w = svc.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_min, x_max)
    yy = a * xx - (svc.intercept_[0]) / w[1]

    plt.figure()
    plt.plot(xx, yy, 'k-')
    # plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=80, facecolors='none')
    plt.scatter(data["X1"], data["X2"], c=data["y"], cmap=plt.cm.Paired)
    plt.axis('tight')
    plt.show()

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

def SVC():
    svc = svm.SVC(C=100, gamma=10, probability=True)
    print (svc)

    data = get_training_data('data/ex6data2.mat')

    svc.fit(data[['X1', 'X2']], data['y'])
    print (svc.score(data[['X1', 'X2']], data['y']))

    data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:,0]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
    
    plt.show()

def serach_best_parameters():
    mat  = loadmat("data/ex6data3.mat")
    training = pd.DataFrame(mat['X'], columns=['X1', 'X2'])
    training['y'] = mat.get('y')

    cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
    cv['y'] = mat.get('yval')
    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    combination = [(C, gamma) for C in candidate for gamma in candidate]
    search = []

    for C, gamma in combination:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(training[['X1', 'X2']], training['y'])
        search.append(svc.score(cv[['X1', 'X2']], cv['y']))

    best_score = search[np.argmax(search)]
    print (np.argmax(search))
    best_param = combination[np.argmax(search)]

    print(best_score, best_param)
    gamma,c = best_param
    best_svc = svm.SVC(C=c, gamma=gamma)
    best_svc.fit(training[['X1', 'X2']], training['y'])
    ypred = best_svc.predict(cv[['X1', 'X2']])

    print(metrics.classification_report(cv['y'], ypred))

def grid_search_cV():
    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    mat  = loadmat("data/ex6data3.mat")
    training = pd.DataFrame(mat['X'], columns=['X1', 'X2'])
    training['y'] = mat.get('y')

    parameters = {'C': candidate, 'gamma': candidate}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    clf.fit(training[['X1', 'X2']], training['y'])
    print (clf.best_params_)

def get_span_data():
    spam_train = loadmat('data/spamTrain.mat')
    spam_test = loadmat('data/spamTest.mat')

    X = spam_train['X']
    Xtest = spam_test['Xtest']
    y = spam_train['y'].ravel()
    ytest = spam_test['ytest'].ravel()

    return X,y,Xtest,ytest

def spam_filter_with_svc():

    X,y,Xtest,ytest = get_span_data()

    svc = svm.SVC()
    svc.fit(X, y)
    print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
    print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))

    pred = svc.predict(Xtest)
    print(metrics.classification_report(ytest, pred))

def spam_filter_with_linear_logistic_regresion():
    # linear logistic regresion
    X,y,Xtest,ytest = get_span_data()
    logit = LogisticRegression()
    logit.fit(X, y)
    pred = logit.predict(Xtest)

    print (metrics.classification_report(ytest, pred))
    print (logit.score(X, y))
    print (logit.score(Xtest, ytest))

######################
def get_min_max(x1,x2):

    x_max =0
    x_min = 0
    
    if min(x1) < min(x2):
        x_min=min(x1)
    else:
        x_min=min(x2)

    if max(x1) > max(x2):
        x_max=max(x1)
    else:
        x_max=max(x2)

    return int(x_min),math.ceil(x_max)

def  test_LinearSVC():
    linear_svc(1)
    linear_svc(100)


def test_gaussian_kernel():
    x1 = np.array([1.0, 2.0, 1.0])
    x2 = np.array([0.0, 4.0, -1.0])
    sigma = 2

    print (gaussian_kernel(x1, x2, sigma))

if __name__ == '__main__':

    # visual_data('data/ex6data2.mat')
    # test_LinearSVC()
    # test_gaussian_kernel()
    # SVC()
    # serach_best_parameters()
    grid_search_cV()
    # spam_filter_with_svc()
    # spam_filter_with_linear_logistic_regresion()