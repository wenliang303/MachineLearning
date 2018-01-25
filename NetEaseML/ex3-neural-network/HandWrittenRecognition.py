#encoding=utf-8
#
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report#这个包是评价报告

data = loadmat('ex3data1.mat')
X = data["X"]
y=data["y"]

def plot_an_image(image):
#     """
#     image : (400,)
#     """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))

def plot_100_image(X):
    """ sample 100 image and show them
    assume the image is square

    X : (5000, 400)
    """
    size = int(np.sqrt(X.shape[1]))

    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))  
            #绘图函数，画100张图片

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):

    theta = np.matrix(theta)
    first = np.multiply(-y , np.log(sigmoid(np.dot(X , theta.T))))
    second = np.multiply((1 - y), np.log(1 - sigmoid(np.dot(X , theta.T))))
    return np.sum(first - second) / (len(X))

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    cost_non_reg=cost(theta, X, y)
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:], 2))
    return cost_non_reg + reg

# def gradient(theta, X, y, learningRate):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
    
#     parameters = int(theta.ravel().shape[1])
#     error = sigmoid(np.dot(X , theta.T)) - y

#     grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
#     # intercept gradient is not regularized
#     grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

#     return np.array(grad).ravel()
    
def gradient(theta, X, y):
    
    m = X.shape[0]
    inner = np.dot(sigmoid(np.dot(theta ,X.T)) - y.T,X)  # (1,m) @ (m, n) -> (1, n)
    return inner / m

def gradientReg(theta, X, y, learningRate):
    lamda = 1
    #  '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = np.multiply((learningRate * lamda/ len(X)) , theta_j1_to_n)

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        
        # minimize the objective function
        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradientReg)
        all_theta[i-1,:] = fmin.x
    
    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)
    
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)
    
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    
    return h_argmax

def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

def forward_predict():

    x = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept
    theta1, theta2 = load_weight('ex3weights.mat')
    print (x.shape, y.shape)
    a1 = x
    z2 = np.dot(a1 , theta1.T) # (5000, 401) @ (25,401).T = (5000, 25)
    print (z2.shape)

    z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
    a2 = sigmoid(z2)
    print (a2.shape)

    z3 = np.dot(a2 , theta2.T)
    print (z3.shape)

    a3 = sigmoid(z3)

    y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行

    print(classification_report(y, y_pred))

##################
def test_plot_one_img():
    pick_one = np.random.randint(0, 5000)
    plot_an_image(X[pick_one, :])
    plt.show()
    print('this should be {}'.format(y[pick_one]))

def test_plot_100_image():
    plot_100_image(X)
    plt.show()

def test_predict():
    print (np.unique(y))#看下有几类标签
    all_theta = one_vs_all(X, y, 10, 1)

    y_pred = predict_all(X, all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print ('accuracy = {0}%'.format(accuracy * 100))

#test_plot_one_img()
#test_plot_100_image()
#get_training_data()
#test_predict()
forward_predict()
