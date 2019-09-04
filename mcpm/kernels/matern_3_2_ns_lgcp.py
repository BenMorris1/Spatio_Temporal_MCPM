import numpy as np
import tensorflow as tf
import sklearn.metrics.pairwise as sk

from mcpm import util
from . import kernel


# This function computes the Matern 3/2 kernel 

class NS_Matern_3_2(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, inducing_inputs, input_dim, lengthscale=0.1, std_dev=1.0, white=0.1, input_scaling=False):
        if input_scaling:
            if lengthscale.size > 1:
                self.lengthscale = tf.Variable(lengthscale)
            else: 
                self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]))
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.white = white
        
        self.N = np.shape(inducing_inputs)[0]
        
        # initialising hyper-parameters
        self.sigma1 = np.zeros(inducing_inputs.shape)
        self.sigma2 = np.zeros(inducing_inputs.shape)
        
        # inputs
        #X = inducing_inputs[:,-1]
        
        #for i in range(self.N):
         #   cov = sk.rbf_kernel(X[i].reshape(1,-1),np.zeros(X[i].reshape(1,-1).shape))
          #  self.sigma1[i] = np.repeat(cov,self.N)
           # self.sigma2[:,i] = np.repeat(cov,self.N)
            
                    
        #self.sigma1 = self.sigma1.astype('float32')
        #self.sigma2 = self.sigma2.astype('float32')
                
        
        
    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0
            
        # find dimension of inputs
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            N1 = sess.run(tf.shape(points1)[0])
            N2 = sess.run(tf.shape(points2)[0])
            n_inputs = sess.run(tf.shape(points1)[1])
            input1 = sess.run(points1)
            input2 = sess.run(points2)
            
        # initialising hyper-parameters
        sigma1 = np.zeros((N1,N2))
        sigma2 = np.zeros((N2,N1))
        #print(sigma1.shape)
        #print(n_inputs)
        
        for i in range(N1):
            cov1 = sk.rbf_kernel(input1[i].reshape(1,-1),np.zeros((1,n_inputs)))
            sigma1[i] = np.repeat(cov1,N2)
        #print(sigma1)    
        
        for j in range(N2):
            cov2 = sk.rbf_kernel(input2[i].reshape(1,-1),np.zeros((1,n_inputs)))
            sigma2[:,i] = np.repeat(cov2,N1)
            
        sigma1 = sigma1.astype('float32')
        sigma2 = sigma2.astype('float32')
        
        # calculating distance matrix
        X = points1 / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), axis=1)
        X2 = points2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        r2 = -2.0 * tf.matmul(X, X2, transpose_b=True)
        r2 += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        r2 = tf.clip_by_value(r2, 0.0, self.MAX_DIST)
        r = tf.sqrt(r2 + 1e-6)
        
        # calculate q matrix
        sigma = tf.add(sigma1,sigma2)/2
        sigma = sigma**(-1)
        q2 = r * sigma * r#tf.multiply(r,tf.multiply(sigma,r))
        q = tf.sqrt(q2)
        
        # sigma term
        sigma_term1 = (tf.multiply(tf.abs(sigma1)**(1/4),tf.abs(sigma2)**(1/4)))
        sigma_term2 = tf.abs(tf.add(sigma1,sigma2)/2)**(-1/2)
        sigma_term = tf.multiply(sigma_term1, sigma_term2)
        
        kernel_matrix = (self.std_dev ** 2) * sigma_term * (1. + np.sqrt(3.) * q) * tf.exp(-np.sqrt(3.) * q)

        #init_op = tf.initialize_all_variables()

        #run the graph
        #with tf.Session() as sess:
         #   sess.run(init_op)
          #  print(sess.run(sigma))
           # print(sess.run(kernel_matrix))
            #print(sess.run(tf.linalg.eigh(kernel_matrix)[0]))
        
        
        return kernel_matrix + white_noise


    def diag_kernel(self, points):
        sigma1 = self.sigma1
        sigma2 = self.sigma2
        sigma_term = (tf.multiply(tf.abs(sigma1)**(1/4),tf.abs(sigma2)**(1/4)))/(tf.abs(tf.add(sigma1,sigma2))**(1/2))
        sigma_diag = tf.matrix_diag_part(sigma_term)
        
        return ((self.std_dev ** 2) * (2**(1/2)) * sigma_diag) + self.white

    def get_params(self):
        return [self.lengthscale, self.std_dev]

