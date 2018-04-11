import tensorflow as tf
import numpy as np


n = 1000
d = 100

iter = 1000
learning_rate = 0.01

x_data = np.vstack([np.random.normal(0.1, 1, (n // 2, d)),
                    np.random.normal(-0.1, 1, (n // 2, d))])
y_data = np.hstack([np.ones(n // 2), -1.0 * np.ones(n // 2)]).reshape((1000, 1))

X = tf.placeholder(tf.float32, [1000, 100])
Y = tf.placeholder(tf.float32, [1000, 1])

w = tf.Variable(tf.random_normal([d, 1], mean=0, stddev=1))

loss = tf.reduce_mean(
    tf.maximum(tf.zeros([1000, 1]),
               tf.ones([1000, 1]) - tf.multiply(Y, tf.matmul(X, w))))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(iter):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(sess.run(loss, feed_dict={X: x_data, Y: y_data}))
