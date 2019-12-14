import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist_data/', one_hot=True)  # 导入mnist数据集

x = tf.placeholder('float', [None, 28, 28, 1])
y = tf.placeholder('float', [None, 10])
keep_prob = tf.placeholder('float')

# 第一层卷积层
w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))  # 第一层卷积层卷积核[5,5,1]，数量32
b_conv1 = tf.Variable(tf.constant(.1, shape=[32]))
conv1 = tf.nn.conv2d(input=x, filter=w_conv1, strides=[1, 1, 1, 1],
                     padding='SAME') + b_conv1  # strides=[1,1,1,1] 四个1分别代表 batchsize x,y,颜色通道数 （个人理解）
# 这里只需要记得中间两个1的意思即可，即沿着行，列每次移动的格子数
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')  # ksize只考虑中间两个参数2，2，表示每次进行池化的矩阵行列大小
# strides部分与conv层类似

# 第二层卷积层
w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))  # 第三个参数要和上一卷积层的卷积核个数一致
b_conv2 = tf.Variable(tf.constant(.1, shape=[64]))
conv2 = tf.nn.conv2d(input=pool1, filter=w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层，卷积层只进行特征的提取，最后还需要全连接层进行分类
w_fc = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=0.1))  # 每一次进行max_pooling操作，图像的行列都要除2，最后就变成7*7
# 最后一层的卷积核有64个所以pool2矩阵维度就是[7,7,64]
b_fc = tf.Variable(tf.constant(.1, shape=[1024]))
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # 将pool2矩阵拉伸成向量，这样才能输入全连接层
fc = tf.matmul(pool2_flat, w_fc) + b_fc
fc = tf.nn.relu(fc)
fc = tf.nn.dropout(fc, keep_prob)

# 接下来是将全连接层的输出转换到标签的10个维度
W = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
B = tf.Variable(tf.constant(.1, shape=[10]))
y_pred = tf.matmul(fc, W) + B

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))  # 指定损失函数
opt = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)  # 指定优化器，去最小化损失函数

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
# 上两行代码是将输出与标签作比较计算出一个精度

# 下面基本是固定模板
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 100
    for step in range(1000):  # 迭代1000次
        batch = mnist.train.next_batch(batch_size)
        batch_input = batch[0].reshape([batch_size, 28, 28, 1])  # 由于mnist数据集每张图像时784的向量，用卷积神经网络时需还原成图像矩阵，即[28,28,1]
        batch_lable = batch[1]
        _, train_loss, acc = sess.run([opt, loss, accuracy], feed_dict={x: batch_input, y: batch_lable, keep_prob: 0.5})
        if (step) % 100 == 0:
            print('step=', step, '  acc=', acc)
