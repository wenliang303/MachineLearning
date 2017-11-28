#encoding=utf-8

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


np.random.seed(1)

def test_init():
    y_hat = tf.constant(36, name="y_hat")
    y = tf.constant(39, name='y')
    loss = tf.Variable((y-y_hat) ** 2, name='loss')

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        print(session.run(loss))

def test_multiply():
    a = tf.constant(3)
    b = tf.constant(30)
    c = tf.multiply(a, b)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        print(session.run(c))

def test_feed():
    sess = tf.Session()
    xx = tf.placeholder(tf.int32, name='any_name')
    res = sess.run(xx * 2, feed_dict={xx:30})
    print (res)
    a = tf.constant(3)
    print(a.__dict__.items())
    sess.close()

#################################
test_init()
test_multiply()
test_feed()
