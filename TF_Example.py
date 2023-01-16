import numpy as np
np.random.seed(0)
import tensorflow as tf

N,D = 3,4
with tf.device('/cpu:0') :
    x = tf.Variable(tf.ones(dtype = tf.float32))
    y = tf.Variable(tf.ones(dtype = tf.float32))
    z = tf.Variable(tf.ones(dtype = tf.float32))

    a = x*y
    b = a+z
    c = tf.reduce_sum(b)

    grad_x, grad_y, grad_z = tf.gradients(c,[x,y,z])

with tf.Session() as sess :
    values = {
        x: np.random.randn(N,D),
        y: np.random.randn(N,D),
        z: np.random.randn(N,D),
    }

    out = sess.run([c,grad_x,grad_y,grad_z],feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out