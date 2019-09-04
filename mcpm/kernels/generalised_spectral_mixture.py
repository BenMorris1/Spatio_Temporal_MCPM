import numpy as np
import tensorflow as tf
import sklearn.metrics.pairwise as sk

from . import Kernel
from .. import util

class GSM(Kernel):
    _id = 0
    def __init__(self, inducing_inputs, num_dimensions=1, num_components = 1, weights = [[1.0]], means = None , var_scale=1.0, 
                 mean_scale=1.0, jitter = 0.01, init=False, mask=None, std_dev=1.0, num_latent=2):
        #super().__init__(mask = mask, **kwargs)
        GSM._id += 1

        self.num_dimensions = num_dimensions
        self.num_components = num_components
        self.means  = means
        self.mean_scale = mean_scale
        self.var_scale = var_scale
        self.weights = weights

        self.white = jitter
        self.std_dev = std_dev
        self.num_latent = num_latent
        self.inducing_inputs = inducing_inputs
        
        n = np.shape(inducing_inputs)[0]
        self.N = n
        
        # initialising hyper-parameters
        self.w1 = np.zeros((n,n))
        self.w2 = np.zeros((n,n))
        self.l1 = np.zeros((n,n))
        self.l2 = np.zeros((n,n))
        self.mu1 = np.zeros((n,n))
        self.mu2 = np.zeros((n,n))
        
        # summing hyper-parameters over latent functions
        for i in range(self.num_latent):
            w1 = np.zeros((n,n))
            w2 = np.zeros((n,n))
            l1 = np.zeros((n,n))
            l2 = np.zeros((n,n))
            mu1 = np.zeros((n,n))
            mu2 = np.zeros((n,n))           
            
            X = inducing_inputs[:,1]
            X = X.reshape(-1,1)
            
            for j in range(n):
                for k in range(n):
                    sigma = sk.rbf_kernel(np.array([X[j]]),np.array([X[k]]))
                    if w1[j,k] == 0:
                        gp_draw = np.random.normal(0,sigma)
                        w1[j,k] = gp_draw
                        w1[k,j] = gp_draw
                    if w2[j,k] == 0:
                        gp_draw = np.random.normal(0,sigma)
                        w2[j,k] = gp_draw
                        w2[k,j] = gp_draw
                    if l1[j,k] == 0:
                        gp_draw = np.random.normal(0,sigma)
                        l1[j,k] = gp_draw
                        l1[k,j] = gp_draw
                    if l2[j,k] == 0:
                        gp_draw = np.random.normal(0,sigma)
                        l2[j,k] = gp_draw
                        l2[k,j] = gp_draw
                    if mu1[j,k] == 0:
                        gp_draw = np.random.normal(0,sigma)
                        mu1[j,k] = gp_draw
                        mu1[k,j] = gp_draw
                    if mu2[j,k] == 0:
                        gp_draw = np.random.normal(0,sigma)
                        mu2[j,k] = gp_draw
                        mu2[k,j] = gp_draw
                        
                        
            # summing hyper-parameters over latent functions
            self.w1 = np.add(self.w1,np.exp(w1))
            self.w2 = np.add(self.w2,np.exp(w2))
        
            self.l1 = np.add(self.l1,np.exp(l1))
            self.l2 = np.add(self.l2,np.exp(l2))
        
            mu1 = np.exp(mu1)/(1+np.exp(mu1))
            self.mu1 = np.add(self.mu1,mu1)
            mu2 = np.exp(mu2)/(1+np.exp(mu2))
            self.mu2 = np.add(self.mu2,mu2)
        

        self.parameters = [self.w1, self.w2, self.l1, self.l2, self.mu1, self.mu2]

    def kernel(self, _X1, _X2, jitter=False, debug=False):
        
                   
        w_1 = self.w1
        w_2 = self.w2
        l_1 = self.l1
        l_2 = self.l2
        mu_1 = self.mu1
        mu_2 = self.mu2
        

        #[1, N1] - [N2, 1] = [N1, N2]
        X1 = tf.transpose(tf.expand_dims(_X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(_X2, perm=[1, 0]), -2)  # D x N2 x 1
        T = tf.transpose(tf.abs(tf.subtract(X1, X2)), perm=[0, 1, 2])  # D x N1 x N2
        T = tf.clip_by_value(T, 0, 1e8)
        T2 = tf.square(T)
        
        #print(np.matmul(np.transpose(w_1),w_2))
        #print(w_1.shape)
    
        # calculating separate parts of kernel equation
        constant_term = (w_1 * w_2 * (np.sqrt((2*l_1*l_2)/(l_1**2 + l_2**2)))).astype('float32')
        cos_term = tf.transpose(tf.abs(tf.subtract(tf.multiply(X1,mu_1), tf.multiply(X2,mu_2))), perm=[0, 1, 2])
        cos_term = tf.clip_by_value(cos_term, 0, 1e8)
        cos_term = cos_term + 1e-6
        exp_term = T2/(l_1**2 + l_2**2)
    
        kern = tf.reshape(tf.multiply(constant_term,tf.multiply(tf.cos(tf.scalar_mul(2*np.pi, cos_term)),
                                                    tf.exp(tf.scalar_mul(-1, exp_term)))),[self.N,self.N])
        
        if jitter is True:
            kern = kern + self.white * tf.eye(tf.shape(_X1)[0])
            

        return kern
    
    
    def diag_kernel(self,points):
        
        w_1 = self.w1
        w_2 = self.w2
        l_1 = self.l1
        l_2 = self.l2
        mu_1 = self.mu1
        mu_2 = self.mu2
        
        # finding diagonal of hyperparameters
        w1_diag = np.diag(w_1)
        w2_diag = np.diag(w_2)
        l1_diag = np.diag(l_1)
        l2_diag = np.diag(l_2)
        mu1_diag = np.diag(mu_1)
        mu2_diag = np.diag(mu_2)
        
        # finding parts of daigonal
        constant_term = w1_diag * w2_diag * (np.sqrt((2*l1_diag*l2_diag)/(l1_diag**2 + l2_diag**2)))
        cos_term = mu1_diag*points - mu2_diag*points
        
        diag_kern = tf.reshape(tf.multiply(constant_term, tf.cos(tf.scalar_mul(2*np.pi, cos_term))),[400])
        
        
        return diag_kern
 
    def get_parameters(self):
        return self.parameters