import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


ITER = 1000
alpha = 0.5

# read input data
mnist = input_data.read_data_sets('mnist_train_data/', one_hot=True)

# placeholder for input data (28 * 28 pixels, None means "can be any length")
x = tf.placeholder(tf.float32, [None, 28 * 28])

W = tf.Variable(tf.zeros([28 * 28, 10]))
b = tf.Variable(tf.zeros([1, 10]))

# this is my model!
y = tf.matmul(x, W) + b

# placeholder for holding correct answer input
y_ = tf.placeholder(tf.float32, [None, 10])

# softmax cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# gradient descent algorithm trainer
trainer = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# let's train it!

for _ in range(ITER):
    batch_xs, bacth_ys = mnist.train.next_batch(100)
    sess.run(trainer, feed_dict={x: batch_xs, y_: bacth_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
