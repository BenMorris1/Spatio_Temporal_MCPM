import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from time import sleep
#import resource

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool
from methods import *
from mcpm.util.generate_data import *
from mcpm.util.process_results import *
from mcpm.util.util import *
from mcpm.util.utilities import *

np.random.seed(1500)

# This code does the following:
# generate the values for f and w
# generate the values y \sim Poisson(exp(w*f + m))
# generate 50 missing obs (for each task) in the outputs and train the model.

# N_all = total number of observations.
# n_missing_values = number of missing obs for each task. 
# n_tasks = number of tasks
# n_latent = number of latent functions used
# sparsity = sparsity in the inputs considering M training points
# inducing_on_inputs = inducing inputs must concide with some training points or not
# num_samples_ell = num of samples to evaluate the ell term.  
# var_steps = variational steps
# epochs= total number of epochs to be optimized for. Epochs are complete passes over the data.
# n_cores = number of cores to use in multiprocessing of single task learning

######### SETTINGS ############################################################
N_all = 400
n_missing_values = 50
n_tasks = 14
num_latent = 2
sparsity = False
sparsity_level = 1.0
inducing_on_inputs = True
optim_ind = True
num_samples_ell = 10
n_sample_prediction = 100
n_bins = 100
epochs=500
var_steps=1 # var set need to be at least one with epochs > 0!!!
display_step_nelbo = 1
inputs_dimension = 3
missing_exp = True
offset_type = 'task' # Specify the type of offset - task-specific or common
trainable_offset = False
n_folds = 1
n_cores = 4
intra_op_parallelism_threads = 0
inter_op_parallelism_threads = 0
input_scaling = False
# Specify the model to be used ("MCPM" of "LGCP")
method = "MCPM"

# Specify if experiment is noisy or not ("noise" or "no_noise")
experiment_type = "noise"

# Specify the quantity to use for predictions (mean, median or mode): 
point_estimate = 'mean'

# Specify the type of prior to use when training MCPM ("Normal" or "GP"):
prior_mixing_weights = "GP"

# Specify the type of Spatio-temporal kernels to use ("Separable_Product" or "Separable_Sum"):
kernel_type = "Separable_Sum"

# Specify the types of kernel to use:
# Spatial Kernel - "Exponential", "Linear", "Matern_3_2", "Matern_5_2", "RadialBasis", or "NS_Matern_3_2"
# Temporal kernel - "Periodic", "SpectralMixture", "GeneralizedSpectralMixture"
spatial_kernel = "NS_Matern_3_2"
temporal_kernel = "Periodic"
if input_scaling == True:
	if method == 'MT':
		num_kernel_hyperpar = num_latent +  (num_latent*inputs_dimension)
	else:
		num_kernel_hyperpar = 1 + inputs_dimension
else:
	if method == 'MT':
		num_kernel_hyperpar = 2*(num_latent)
	else:
		num_kernel_hyperpar = 2


######### IMPORT DATA  #####################################################
        
# import data as csv
file_path = os.path.join(os.path.join(os.path.dirname(__file__), ".."),'Data','UK_Crime_Data','data_final.csv')
data = pd.read_csv(file_path)
        
# imputs
inputs = data[['Longitude','Latitude','Time']].values

# defining the original inputs
original_inputs = inputs

# converting longitude and latitude to NAD83 and creating inputs
inputs = LatLong_NAD83(inputs)
#inputs = np.concatenate((space_inputs,inputs[:,2].reshape(inputs.shape[0],1)),axis=1)

# outputs 
outputs = data[['Anti-social behaviour','Bicycle theft','Burglary','Criminal damage and arson',
                'Drugs','Other crime','Other theft','Possession of weapons','Public order',
                'Robbery','Shoplifting','Theft from the person','Vehicle crime','Violence and sexual offences']].values
np.save(os.path.join(os.path.dirname(__file__), "..") + '/Data/UK_Crime_Data/' + kernel_type + '/outputs', outputs)

# generate task features
task_features = get_features(outputs)

# generating missing data
ytrain_non_missing_index = generate_missing_data_uk_crime(outputs, missing_exp)

# saving data
np.save(os.path.join(os.path.dirname(__file__), "..") + '/Data/UK_Crime_Data/' + kernel_type + '/original_inputs', original_inputs)
np.save(os.path.join(os.path.dirname(__file__), "..") + '/Data/UK_Crime_Data/' + kernel_type + '/inputs', inputs)
np.save(os.path.join(os.path.dirname(__file__), "..") + '/Data/UK_Crime_Data/' + kernel_type + '/task_features', task_features)
np.save(os.path.join(os.path.dirname(__file__), "..") + '/Data/UK_Crime_Data/' + kernel_type + '/outputs_missing', ytrain_non_missing_index)

# Define the inputs for training and testing
xtrain = inputs
ytrain = outputs
xtest = inputs

# Determine the number of testing points and training points. 
# In the synthetic experiment they are both equal to N_all. 
num_train = xtrain.shape[0]
num_test = xtest.shape[0]


######### INITIALISATION ######################################################
# Initialise kernel hyperpars and lik pars
lengthscale_initial = np.float32(0.05)#(1.0)
sigma_initial = np.float32(0.1)

if offset_type == 'task':
    offset_initial = np.float32(np.repeat(2.0, n_tasks))[:,np.newaxis]
	#offset_initial = np.float32(offset_data + random_noise)[:,np.newaxis]
else:
    offset_initial = np.float32(2.0)[:,np.newaxis]
	#offset_initial = np.float32(offset_data + random_noise)[:,np.newaxis][0]

# Initialize the kernel hyperparameters for the weight processes
lengthscale_initial_weights = np.float32(0.2)
sigma_initial_weights = np.float32(1.0)

# Set the white noise needed for the inversion of the kernel
white_noise = 0.01


######## TRAINING #############################################################

if method == 'MCPM':
	print('I am doing MCPM')
	(fold, pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets,
	nelbo_values, time_iterations) = MCPM_ST_learning(xtrain, xtest, ytrain, task_features, kernel_type, spatial_kernel, temporal_kernel,
																						prior_mixing_weights, point_estimate, ytrain_non_missing_index, 
																						n_missing_values, sparsity, sparsity_level, inducing_on_inputs, optim_ind, 
																						offset_type, offset_initial, n_tasks, 
																						num_latent, trainable_offset, lengthscale_initial, sigma_initial, 
																						white_noise, input_scaling, lengthscale_initial_weights, 
																						sigma_initial_weights, prior_mixing_weights, num_samples_ell, 
																						epochs, var_steps, display_step_nelbo, intra_op_parallelism_threads, 
																						inter_op_parallelism_threads)


	pred_mean = pred_mean[np.newaxis]
	pred_var = pred_var[np.newaxis]
	covars_weights = np.concatenate(covars_weights, axis=0)
	means_w = np.concatenate(means_w, axis=0)
	offsets = np.concatenate(offsets, axis=0)


if method == 'LGCP':
	print('I am doing LGCP')
	def Full_LGCP_learning(task):
	#for task in range(n_tasks):
		return LGCP_ST_learning(xtrain, xtest, ytrain, task_features, kernel_type, spatial_kernel, temporal_kernel, point_estimate, ytrain_non_missing_index, sparsity, sparsity_level, 
                       inducing_on_inputs, optim_ind, offset_type, trainable_offset, lengthscale_initial, sigma_initial, white_noise, input_scaling, 
                       lengthscale_initial_weights, sigma_initial_weights, prior_mixing_weights, num_samples_ell, epochs, var_steps, display_step_nelbo, 
                       intra_op_parallelism_threads, inter_op_parallelism_threads, task)


	task_list = list(range(0,n_tasks,1))
	pool = Pool(processes = n_cores)
	results_single_task_loop = pool.map(Full_LGCP_learning, task_list)	

	## Process results
	# This function create tensors where to store the values for each task when using ST
	# It extracts results from the multiprocessing output assigning them to the corresponding tensors
	(pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, 
	time_iterations, nelbo_values) = post_process_results_LGCP(results_single_task_loop, N_all, n_tasks, num_latent, num_train, num_test, epochs, display_step_nelbo, 
                              num_kernel_hyperpar, n_missing_values, sparsity_level, n_folds, inputs_dimension, method, prior_mixing_weights)




######### SAVING RESULTS ######################################################
folder = os.path.join(os.path.dirname(__file__), "..") + '/Data/UK_Crime_Data/' + kernel_type + '/'
suffix = prior_mixing_weights + "_" + method + "_" + str(missing_exp)
suffix2 = prior_mixing_weights + "_" + method + "_" + str(missing_exp) + str(num_samples_ell)

#if experiment_type == "noise":
#    np.save(folder + 'random_noise_vector', random_noise_vector)
    
# Create a dataset with data and predictions and save it 
final_dataset = np.zeros((n_folds, num_train, (n_tasks*3 + inputs_dimension)))
for i in range(n_folds):
		final_dataset[i] = np.concatenate((inputs, outputs, pred_mean[i], pred_var[i]), axis = 1)
np.save(folder + 'final_dataset_' + suffix, final_dataset)

# Save kernel info
# np.save(folder + 'kernel_params_final_' + prior_mixing_weights + "_" + method + "_" + str(missing_exp), kernel_params_final)
# np.save(folder + 'kernel_params_initial_' + prior_mixing_weights + "_" + method + "_" + str(missing_exp), kernel_params_initial)


# Save nelbo values, time iterations and variables' values over epochs
np.save(folder + 'nelbo_values_' + suffix2, nelbo_values)
np.save(folder + 'time_iterations_' + suffix2, time_iterations)
# np.save(folder + 'f_mu_' + suffix2, f_mu_tensor)
# np.save(folder + 'f_var_' + suffix2, f_var_tensor)
# np.save(folder + 'w_mean_' + suffix2, w_mean_tensor)
# np.save(folder + 'w_var_' + suffix2, w_var_tensor)
# np.save(folder + 'off_' + suffix2, off_tensor)


# Save latent functions and weights info
np.save(folder + 'latent_means_' + suffix, latent_means)
np.save(folder + 'latent_variances_' + suffix, latent_vars)
np.save(folder + 'means_weights_' + suffix, means_w)
np.save(folder + 'covars_weights_' + suffix, covars_weights)
np.save(folder + 'offsets_' + suffix, offsets)