import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
X_holder = tf.placeholder(tf.float32)
y_holder = tf.placeholder(tf.float32)

def addConnect(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)

# ==========================================================
# 网络层数过多，增加模型复杂度，更难收敛（过拟合或陷入局部最优）
# connect_1 = addConnect(X_holder, 784, 300, tf.nn.relu)
# connect_2 = addConnect(connect_1, 300, 100, tf.nn.relu)
# connect_3 = addConnect(connect_2, 100, 50, tf.nn.relu)
# predict_y = addConnect(connect_3, 50, 10, tf.nn.softmax)
# ==========================================================
connect_1 = addConnect(X_holder, 784, 300, tf.nn.relu)
predict_y = addConnect(connect_1, 300, 10, tf.nn.softmax)
# ==========================================================
# predict_y = addConnect(X_holder, 784, 10, tf.nn.softmax)
# ==========================================================
loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
optimizer = tf.train.AdagradOptimizer(0.3)
train = optimizer.minimize(loss)

session = tf.Session()

# 开始训练
# init = tf.global_variables_initializer()
# session.run(init)
#
# for i in range(1000):
#     images, labels = mnist.train.next_batch(batch_size)
#     session.run(train, feed_dict={X_holder:images, y_holder:labels})
#     if i % 100 == 0:
#         correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
#         print('step:%d accuracy:%.4f' %(i, accuracy_value))

# 保存模型
# saver = tf.train.Saver(tf.global_variables())
# saver.save(session, 'mnist_dnn_model/model')
# 读取模型
saver = tf.train.Saver(tf.global_variables())
traind_model = saver.restore(session, 'mnist_dnn_model/model')

# 模型推断输出结果
test_data_x = mnist.test.images[0].reshape(-1,784)
test_data_label = mnist.test.labels[0]
print( test_data_x, test_data_label)
print( session.run(tf.argmax(predict_y,1), feed_dict={X_holder: test_data_x}))
