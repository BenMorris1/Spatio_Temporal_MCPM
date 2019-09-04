import numpy as np
import tensorflow as tf

from . import util


class Normal(object):
    def __init__(self, mean, covar):
        self.mean = mean
        self.covar = covar


class CholNormal(Normal):
    def prob(self, val):
        return tf.exp(self.log_prob(val))

    def log_prob(self, val):
        dim = tf.cast(tf.shape(self.mean)[0], dtype=tf.float32)
        diff = tf.expand_dims(val - self.mean, 1)
        quad_form = tf.reduce_sum(diff * tf.linalg.cholesky_solve(self.covar, diff))
        return -0.5 * (dim * tf.math.log(2.0 * np.pi) + util.log_cholesky_det(self.covar) +
                       quad_form)


class DiagNormal(Normal):
    def prob(self, val):
        return tf.exp(self.log_prob(val))

    def log_prob(self, val):
        dim = tf.cast(tf.shape(self.mean)[0], dtype=tf.float32)
        #quad_form = tf.reduce_sum(self.covar * (val - self.mean) ** 2)
        quad_form = tf.reduce_sum((val - self.mean) ** 2 / self.covar)
        return -0.5 * (dim * tf.math.log(2.0 * np.pi) + tf.reduce_sum(tf.math.log(self.covar)) + quad_form)

