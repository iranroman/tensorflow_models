"""Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_tr = np.load('../../../../experiments/tr_data/x_tr.npy')[:,:,10:34]
X_tr = np.delete(X_tr,[32,42,23,31,41,33,59,63,14,22],axis=1)
X_tr = 1e6*X_tr
X_tr = X_tr.astype('float32')
Y_tr = np.load('../../../../experiments/tr_data/y_tr.npy')
X_ts = np.load('../../../../experiments/ts_data/x_ts.npy')[:,:,10:34]
X_ts = np.delete(X_ts,[32,42,23,31,41,33,59,63,14,22],axis=1)
X_ts = 1e6*X_ts
X_ts = X_ts.astype('float32')
Y_ts = np.load('../../../../experiments/ts_data/y_ts.npy')


def separate_tr_labels(X_tr, Y_tr):
	Y_tr_d = np.array([np.where(i==1)[0][0] for i in Y_tr]) 
	idx_vis = np.where(Y_tr_d==0)[0]
	idx_pul = np.where(Y_tr_d==1)[0]
	X_tr_0 = X_tr[idx_vis]
	Y_tr_0 = Y_tr[idx_vis]
	X_tr_1 = X_tr[idx_pul]
	Y_tr_1 = Y_tr[idx_pul]

	return X_tr_0, Y_tr_0, X_tr_1, Y_tr_1

def data_preprocess(images,labels):

	images = np.expand_dims(images, axis=2)
	# get data indices of interest for this specific experiment
	idx_9 = np.nonzero(np.in1d(labels[:,0],[2]))[0]
	images = images[idx_9]
	labels = labels[idx_9]
	idx_vis = np.nonzero(np.in1d(labels[:,1],[0]))[0]
	idx_ini = np.nonzero(np.in1d(labels[:,1],[1]))[0]
	idx_ant = np.nonzero(np.in1d(labels[:,1],[2]))[0]
	idx_cha = np.nonzero(np.in1d(labels[:,1],[3]))[0]
	idx_pul = np.concatenate((idx_ini,idx_ant,idx_cha),axis=0)
	labels[idx_vis,1] = 0
	labels[idx_pul,1] = 1
	labels_dense = labels[:,1].astype(np.int)
	labels = np.zeros((labels_dense.size, labels_dense.max()+1))
	labels[np.arange(labels_dense.size),labels_dense] = 1
	
	return images, labels

X_ts, Y_ts = data_preprocess(X_ts,Y_ts)
X_tr, Y_tr = data_preprocess(X_tr,Y_tr)
X_tr_0, Y_tr_0, X_tr_1, Y_tr_1 = separate_tr_labels(X_tr,Y_tr)

# Training Parameters
l2_w = 0.005
learning_rate = 0.0001
num_steps = 20000
batch_size = 128
display_step = 10

# Network Parameters
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, X_ts.shape[1], X_ts.shape[2], X_ts.shape[3]])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, 1, strides, strides], padding='SAME', data_format="NCHW")
    x = tf.nn.bias_add(x, b, data_format='NCHW')
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, 1, 1, k], strides=[1, 1, 1, k],
                          padding='SAME', data_format='NCHW')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.sqrt(2.0/54)*tf.random_normal([1, 5, 54, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.sqrt(2.0/32)*tf.random_normal([1, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.sqrt(2.0/(6*64))*tf.random_normal([1*6*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.sqrt(2.0/(1024))*tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y)) +
    l2_w*tf.nn.l2_loss(weights['wc1']) +
    l2_w*tf.nn.l2_loss(weights['wc2']) +
    l2_w*tf.nn.l2_loss(weights['wd1']) +
    l2_w*tf.nn.l2_loss(weights['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def next_batch(batch_size,X0,Y0,X1,Y1,i):
	n_vis = X0.shape[0]
	n_aud = X1.shape[0]
	n_vis_in_aud = int(n_aud/n_vis)
	i = n_vis_in_aud%i
	i -= 1 	
	data = np.concatenate((X0,X1[i*n_vis:i*n_vis+n_vis]),axis=0) 
	labels = np.concatenate((Y0,Y1[i*n_vis:i*n_vis+n_vis]),axis=0)
	"""
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:batch_size]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]
	"""
	return data, labels
			
	
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = next_batch(batch_size, X_tr_0, Y_tr_0, X_tr_1, Y_tr_1, step)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            # Calculate accuracy for 256 MNIST test images
            print("Testing Accuracy:", \
               	sess.run(accuracy, feed_dict={X: X_ts,
         	      Y: Y_ts,
                      keep_prob: 1.0}))
