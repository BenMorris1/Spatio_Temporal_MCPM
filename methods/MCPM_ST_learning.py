import mcpm
import numpy as np
import tensorflow as tf
import time
import math
import csv
from multiprocessing import Pool
from initialization_inducing import initialize_inducing_points



def MCPM_ST_learning(xtrain, xtest, ytrain, task_features, kernel_type, spatial_kernel, temporal_kernel, prior_weights, point_estimate, 
                     ytrain_non_missing_index, n_missing_values, sparsity, sparsity_level, inducing_on_inputs, optim_ind, offset_type, 
                     offset_initial, n_tasks, num_latent, trainable_offset, lengthscale_initial, sigma_initial, white_noise, input_scaling, 
                     lengthscale_initial_weights, sigma_initial_weights, prior_mixing_weights, num_samples_ell, epochs, var_steps, 
                     display_step_nelbo, intra_op_parallelism_threads, inter_op_parallelism_threads, fold = 0, num_features = 2):


	data = mcpm.datasets.DataSet(xtrain, ytrain)

	N_all = xtrain.shape[0]


	# Initialize the likelihood function.
	if prior_weights == "Normal":
		likelihood = mcpm.likelihoods.Lik_MCPM_N(ytrain_non_missing_index = ytrain_non_missing_index,
													num_missing_data = n_missing_values,
													offset_type = offset_type, offsets = offset_initial, 
													num_tasks = n_tasks, point_estimate = point_estimate, trainable_offset = False)

	if prior_weights == "GP":
		likelihood = mcpm.likelihoods.Lik_MCPM_GP(ytrain_non_missing_index = ytrain_non_missing_index,
													num_missing_data = n_missing_values, offset_type = offset_type, offsets = offset_initial, 
													num_tasks = n_tasks, num_latent = num_latent, trainable_offset = False)

	# Get the dimension of the input
	dim_inputs = xtrain.shape[1]
	num_train = xtrain.shape[0]
	num_test = xtest.shape[0]

	if prior_weights == 'Normal':
		weights = mcpm.Prior_w.Normal(num_tasks = n_tasks, num_latent = num_latent)
	if prior_weights == 'GP':
		weights = mcpm.Prior_w.GP(num_tasks = n_tasks, num_latent = num_latent)
        
    # Define the levels of sparsity we want to train the model 
	# and initialise the inducing inputs (Optimisation of the inducing inputs is only introduced when using sparsity.)
	if sparsity == False:
		sparsity_vector = np.array([1.0])
		inducing_inputs = xtrain 
	else:
		sparsity_vector = np.array([sparsity_level])
		inducing_number = int(sparsity_level*N_all)
		inducing_inputs, _ = initialize_inducing_points(xtrain,ytrain,inducing_on_inputs,num_latent,inducing_number,N_all,dim_inputs)

	# Initiliaze the kernels
	if kernel_type == "Separable_Product":
		kernel = [mcpm.kernels.Separable_Product(dim_inputs, inducing_inputs, spatial_kernel, temporal_kernel, num_latent = num_latent, lengthscale = lengthscale_initial, 
                                           std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling, period = 1.0,
                                           num_dimensions=1, num_components=1, weights=[[1.0]], means=None, var_scale=1.0, mean_scale=1.0, 
                                           init=False, mask=None) for i in range(num_latent)]        
	if kernel_type == "Separable_Sum":
		kernel = [mcpm.kernels.Separable_Sum(dim_inputs, inducing_inputs, spatial_kernel, temporal_kernel, num_latent = num_latent, lengthscale = lengthscale_initial, 
                                       std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling, period = 1.0,
                                       num_dimensions=1, num_components=1, weights=[[1.0]], means=None, var_scale=1.0, mean_scale=1.0, 
                                           init=False, mask=None) for i in range(num_latent)]
    
    
    # Initialise the PRIOR variances for the weights. The prior mean is set to zero. 
	# The prior for the weights need to be given in a format PQX1. We initialise to 1 the variance for all the weights.
	prior_var_w_vector = np.ones(n_tasks*len(kernel), dtype=np.float32)

	# Initialise the kernel for the GPs on the weights. Needed is prior = GP.
	kernel_weights = [mcpm.kernels.RadialBasis(num_features, lengthscale = lengthscale_initial_weights, std_dev = sigma_initial_weights, white = white_noise) for i in range(num_latent)] 



	# Define the model
	model = mcpm.mcpm(likelihood, kernel, weights, inducing_inputs, missing_data = n_missing_values, num_training_obs = num_train, 
							num_testing_obs = num_test, ytrain_non_missing_index = ytrain_non_missing_index, offset_type = offset_type, 
							prior_var_w_vector = prior_var_w_vector, kernel_funcs_weights = kernel_weights, task_features = task_features, 
							num_tasks = n_tasks, optimization_inducing = optim_ind, num_samples_ell = num_samples_ell,
							intra_op_parallelism_threads = intra_op_parallelism_threads, inter_op_parallelism_threads = inter_op_parallelism_threads)

  


	# Define the tf  optimizer
    #optimizer = tf.train.AdamOptimizer(0.005) # all other stationary experiments
	optimizer = tf.compat.v1.train.AdamOptimizer(0.0025)#optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00175) #(separable sum (M52+ SM), normal & GP mixing weights)
	#optimizer=tf.train.AdadeltaOptimizer(0.25)#optimizer = tf.train.AdagradOptimizer(0.005) #(separable sum (M52+ SM), GP mixing weights)
    #optimizer = tf.train.AdadeltaOptimizer(0.005)

	# Start the training of the model
	start = time.time()
    
        
	
    # Train
	(nelbo_values, time_iterations) = model.fit(data, optimizer, var_steps=var_steps, epochs=epochs, display_step=1, display_step_nelbo = display_step_nelbo)


	end = time.time()
	time_elapsed = end-start

	print("Total training finished in seconds", time_elapsed)

	# Predictions
	pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets = model.predict(xtest)

	# Alternative point estimates
	if point_estimate == 'median':
		median_predictions = empirical_median(N_all, n_tasks, num_latent, latent_means, latent_vars, means_w, var_w, offsets, n_bins, n_sample_prediction)
	if point_estimate == 'mode':
		mode_predictions = empirical_mode(N_all, n_tasks, num_latent, latent_means, latent_vars, means_w, var_w, offsets, n_bins)

	return (fold, pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets,
			nelbo_values, time_iterations)


