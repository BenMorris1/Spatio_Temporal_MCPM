import numpy as np
import tensorflow as tf

from mcpm import kernels
from . import kernel
from mcpm.util import *


class Separable_Product(kernel.Kernel):
    MAX_DIST = 1e8
    
    def __init__(self, input_dim, spatial_kernel, temporal_kernel, lengthscale=1.0, std_dev=1.0, white=0.01, input_scaling=False, period=1.0):
        
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]), name = 'lenghtscale', trainable=False)
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32, name = 'lenghtscale', trainable=False)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32, name = 'std_dev', trainable=False)
        self.white = white
        self.input_dim = input_dim
        self.input_scaling = input_scaling
        self.period = period
        
        
        # initialising the spatial kernel
        if spatial_kernel == "Exponential":
            self.spatial_kernel = kernels.Exponential(self.input_dim, self.lengthscale, self.std_dev, self.white, 
                                                            self.input_scaling)
        if spatial_kernel == "RadialBasis":
            self.spatial_kernel = kernels.RadialBasis(self.input_dim, self.lengthscale, self.std_dev, self.white, 
                                                            self.input_scaling)
        if spatial_kernel == "Matern_3_2":
            self.spatial_kernel = kernels.Matern_3_2(self.input_dim, self.lengthscale, self.std_dev, self.white, 
                                                            self.input_scaling)
        if spatial_kernel == "Matern_5_2":
            self.spatial_kernel = kernels.Matern_5_2(self.input_dim, self.lengthscale, self.std_dev, self.white, 
                                                            self.input_scaling)
        #if spatial_kernel == "Linear":
        #    self.spatial_kernel = [mcpm.kernels.Linear(dim_inputs, self.lengthscale = lengthscale_initial, 
        #                                                    self.std_dev = sigma_initial, self.white = white_noise, 
        #                                                    self.input_scaling = input_scaling) for i in range(self.num_latent)]
        
        #initialise the temporal kernel
        if temporal_kernel == "Periodic":
            self.temporal_kernel = kernels.Periodic(self.period, self.std_dev, self.lengthscale, 
                                                            self.white)
        #if spatial_kernel == "SpectralMixture":
        #    self.temporal_kernel = [mcpm.kernels.SM(dim_inputs, self.lengthscale = lengthscale_initial, 
        #                                                    self.std_dev = sigma_initial, self.white = white_noise, 
        #                                                    self.input_scaling = input_scaling) for i in range(self.num_latent)]
        
            
    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0 
        
        # define spatial and temporal element of data
        points1_spatial = points1[:,0]
        points1_spatial,points1_index = tf.unique(points1_spatial)
        points1_spatial = tf.reshape(points1_spatial,[20,1])
        points2_spatial = points2[:,0]
        points2_spatial,points2_index = tf.unique(points2_spatial)
        points2_spatial = tf.reshape(points2_spatial,[20,1])
        
        points1_temporal = points1[:,1]
        points1_temporal,points1_index = tf.unique(points1_temporal)
        points1_temporal = tf.reshape(points1_temporal,[10,1])
        points2_temporal = points2[:,1]
        points2_temporal,points2_index = tf.unique(points2_temporal)
        points2_temporal = tf.reshape(points2_temporal,[10,1])
        
        # define kernels
        spatial_kern = self.spatial_kernel.kernel(points1_spatial,points2_spatial)
        temporal_kern = self.temporal_kernel.kernel(points1_temporal,points2_temporal)
        init_op = tf.initialize_all_variables()
        
        print('spatial')
        with tf.Session() as sess:
            sess.run(init_op)
            #print(sess.run(tf.linalg.eigh(spatial_kern)[0]))
        
        print('temporal')
        with tf.Session() as sess:
            sess.run(init_op)
            #print(sess.run(tf.linalg.eigh(temporal_kern)[0]))
        
        # calculate the kronecker product of kernels
        spatial_operator = tf.linalg.LinearOperatorFullMatrix(spatial_kern)
        temporal_operator = tf.linalg.LinearOperatorFullMatrix(temporal_kern)
        kern = tf.linalg.LinearOperatorKronecker([spatial_operator,temporal_operator]).to_dense()
        
        print('kernel')
        with tf.Session() as sess:
            sess.run(init_op)
            #print(sess.run(tf.linalg.eigh(kern)[0]))
        
        
        return kern + white_noise
    
    
    def diag_kernel(self, points):
        
        # define spatial and temporal points
        points_spatial = points[:,0]
        points_spatial, spatial_index = tf.unique(points_spatial)
        points_spatial = tf.reshape(points_spatial, [20,1])
        points_temporal = points[:,1]
        points_temporal, temporal_index = tf.unique(points_temporal)
        points_temporal = tf.reshape(points_temporal, [10,1])
        
        # define diagonal kernels
        spatial_diag_kern = self.spatial_kernel.diag_kernel(points_spatial)
        temporal_diag_kern = self.temporal_kernel.diag_kernel(points_temporal)
        
        # reshaping kernels
        spatial_diag_kern = tf.reshape(spatial_diag_kern,[20,1])
        temporal_diag_kern = tf.reshape(temporal_diag_kern,[10,1])
        
        
        # find kronecker product of the kernels
        spatial_operator = tf.linalg.LinearOperatorFullMatrix(spatial_diag_kern)
        temporal_operator = tf.linalg.LinearOperatorFullMatrix(temporal_diag_kern)
        diag_kern = tf.linalg.LinearOperatorKronecker([spatial_operator,temporal_operator]).to_dense()
        diag_kern = tf.reshape(diag_kern,[200])
        
        return diag_kern
    

    def get_params(self):
        return [self.lengthscale, self.std_dev]

        
        