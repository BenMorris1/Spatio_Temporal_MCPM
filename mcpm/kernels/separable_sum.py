import numpy as np
import tensorflow as tf

from mcpm import kernels
from . import kernel
from mcpm.util import *


class Separable_Sum(kernel.Kernel):
    MAX_DIST = 1e8
    
    def __init__(self, input_dim, inducing_inputs, spatial_kernel, temporal_kernel, num_latent, lengthscale=1.0, std_dev=1.0, white=0.01, input_scaling=False, period=1.0,
                 num_dimensions=1, num_components=1, weights=[[1.0]], means=None, var_scale=1.0, mean_scale=1.0, init=False, mask=None):
        
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]), name = 'lenghtscale', trainable=False)
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32, name = 'lenghtscale', trainable=False)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32, name = 'std_dev', trainable=False)
        self.white = white
        self.input_dim = input_dim
        self.input_scaling = input_scaling
        self.period = period
        self.num_dimensions = num_dimensions
        self.num_components = num_components
        self.weights = weights
        self.means = means
        self.var_scale = var_scale
        self.mean_scale = mean_scale
        self.init = init
        self.mask = mask
        self.num_latent = num_latent
        self.inducing_inputs = inducing_inputs
        
        
        
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
        if spatial_kernel == "Linear":
            self.spatial_kernel = kernels.Linear(self.input_dim, self.std_dev, self.white)
            
        if spatial_kernel == "NS_Matern_3_2":
            self.spatial_kernel = kernels.NS_Matern_3_2(self.inducing_inputs, self.input_dim, self.lengthscale,
                                                        self.std_dev, self.white, self.input_scaling)
        
        #initialise the temporal kernel
        if temporal_kernel == "Periodic":
            self.temporal_kernel = kernels.Periodic(self.period, self.std_dev, self.lengthscale, self.white)
            
        if temporal_kernel == "SpectralMixture":
            self.temporal_kernel = kernels.SM(self.num_dimensions, self.num_components, self.weights, self.means,
                                              self.var_scale, self.mean_scale, self.white, self.init, self.mask,
                                              self.std_dev)
        
            
    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0 
        
        # define spatial and temporal element of data
        points1_spatial = points1[:,:-1]#[:,np.newaxis]
        points2_spatial = points2[:,:-1]#[:,np.newaxis]
        
        points1_temporal = points1[:,-1][:,np.newaxis]
        points2_temporal = points2[:,-1][:,np.newaxis]
        
        # define kernels
        spatial_kern = self.spatial_kernel.kernel(points1_spatial,points2_spatial)
        temporal_kern = self.temporal_kernel.kernel(points1_temporal,points2_temporal)
        
        kern = tf.math.add(spatial_kern,temporal_kern)
        
        return kern + white_noise
    
    
    def diag_kernel(self, points):
        # define spatial and temporal points
        points_spatial = points[:,0][:,np.newaxis]
        points_temporal = points[:,1][:,np.newaxis]
        
        # define diagonal kernels
        spatial_diag_kern = self.spatial_kernel.diag_kernel(points_spatial)
        temporal_diag_kern = self.temporal_kernel.diag_kernel(points_temporal)
        
        # reshaping kernels
        #spatial_diag_kern = tf.reshape(spatial_diag_kern,[tf.shape(points)[0],1])
        #temporal_diag_kern = tf.reshape(temporal_diag_kern,[tf.shape(points)[0],1])
        
        # find product of the kernels
        #diag_kern = tf.matmul(spatial_diag_kern,tf.transpose(temporal_diag_kern))
        diag_kern = tf.math.add(spatial_diag_kern,temporal_diag_kern)
        diag_kern = tf.reshape(diag_kern,[tf.shape(points)[0]])
        
        return diag_kern
    

    def get_params(self):
        return [self.lengthscale, self.std_dev]

        
        