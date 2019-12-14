import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

class ModelCore(object):
    def __init__(self):
        cwd = os.getcwd()
        self.X_holder = tf.placeholder(tf.float32)
        self.y_holder = tf.placeholder(tf.float32)

        connect_1 = self.addConnect(self.X_holder, 784, 300, tf.nn.relu)
        self.predict_y = self.addConnect(connect_1, 300, 10, tf.nn.softmax)

        loss = tf.reduce_mean(-tf.reduce_sum(self.y_holder * tf.log(self.predict_y), 1))
        optimizer = tf.train.AdagradOptimizer(0.3)
        self.train = optimizer.minimize(loss)
        self.session = tf.Session()

        self.saver = tf.train.Saver(tf.global_variables())
        traind_model = self.saver.restore(self.session, cwd + '/core/model_data/model')

    def addConnect(self, inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01))
        biases = tf.Variable(tf.zeros([1, out_size]))
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            return Wx_plus_b
        else:
            return activation_function(Wx_plus_b)

    def classify(self, image_list):
        x_data = np.array(image_list).reshape(-1,784)
        return self.session.run(tf.argmax(self.predict_y, 1), feed_dict={self.X_holder: x_data})
