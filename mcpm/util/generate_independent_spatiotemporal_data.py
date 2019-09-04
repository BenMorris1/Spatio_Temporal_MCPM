import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mcpm.util.util import init_list
import sklearn
import sklearn.metrics.pairwise as sk
import sklearn.gaussian_process.kernels as sg
from mcpm.util.utilities import *
from mcpm.kernels import *

#import matplotlib.pyplot as plt

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def generate_synthetic_data(N_all, n_tasks, kernel, num_latent, num_features = 2):

	# Initialise required variables
    weights_data_task = init_list(0.0, [n_tasks])
    process_values = init_list(0.0, [num_latent])
    sample_intensity = init_list(0.0, [n_tasks])
    outputs = np.ones((N_all,n_tasks))
    task_features = np.zeros((n_tasks,num_features))

    
    # Define the tasks specific offsets. 
    offset_data = np.float32(np.array([2.0, 2.0, 2.6, 2.6]))

	# Select some weights to generate the data
    weights_data_task[0] = np.float32(np.array([+0.1,-0.12]))
    weights_data_task[1] = np.float32(np.array([-0.1,-0.1])) 
    weights_data_task[2] = np.float32(np.array([-0.1,+0.1]))
    weights_data_task[3] = np.float32(np.array([-0.2,+0.1]))

	# Random noise added to the true parameters in order to initialise the algorithm pars
    random_noise = np.random.normal(loc=0.0, scale=1.0, size=1)

	# Initiliaze the inputs and stardardize them
    space_x = np.linspace(0.0,1.0,N_all/20)
    time_x = np.linspace(0.0,9.0,20)
    space_x,time_x = np.meshgrid(space_x,time_x)
    inputs = np.array([space_x.flatten(),time_x.flatten()]).T
    np.save(os.path.join(os.path.dirname(__file__), "....") + '/Data/Synthetic_Experiments/' + kernel + '/original_inputs_synthetic', inputs)
    inputs_mean = np.transpose(np.mean(inputs))
    inputs_std = np.transpose(np.std(inputs))
    standard_inputs = (inputs - inputs_mean)/inputs_std
    inputs = standard_inputs
    
    # Run the spatial and temporal kernels on the input data
    sigma1_spatial = 6 * sk.rbf_kernel(inputs[:,0][:, np.newaxis], inputs[:,0][:, np.newaxis], gamma=25)
    sigma2_spatial = 6 * sk.rbf_kernel(inputs[:,0][:, np.newaxis], inputs[:,0][:, np.newaxis], gamma=20)
    sigma1_temporal = 6 * sk.linear_kernel(inputs[:,1][:, np.newaxis],inputs[:,1][:, np.newaxis])
    sigma2_temporal = 6 * sk.linear_kernel(inputs[:,1][:, np.newaxis],inputs[:,1][:, np.newaxis])
    sigma1 = np.multiply(sigma1_spatial,sigma1_temporal)
    sigma2 = np.multiply(sigma2_spatial,sigma2_temporal)
    
    # Sample the true underlying GPs. 
    for i in range(num_latent):
        if i == 0:
            np.random.seed(10)
            process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma1)
            process_values[i] = np.reshape(process_values[i], (N_all,1))
        if i == 1:
            process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma2)
            process_values[i] = np.reshape(process_values[i], (N_all,1))

	# Generate the intensities as a linear combination of the latent functions with the weights specific to the task + offset specific to the task
    for i in range(n_tasks):
        weighted_sum = 0.0
        for j in range(num_latent):
            process_values_single = np.array(process_values[j])
            weights_data_task_single = np.array(weights_data_task[i])[:,np.newaxis]
            weighted_sum += weights_data_task_single[j,:]*process_values_single
        sample_intensity[i] = np.exp(weighted_sum + offset_data[i])

	# Generate the outputs by sampling from a Poisson with the constructed intensities
    for j in range(n_tasks):
        for i in range(N_all):
            sample_intensity_single = sample_intensity[j]
            outputs[i,j] = np.random.poisson(lam = sample_intensity_single[i,0])

	# Define some task_features used when placing a GP prior on the mixing weights
    for i in range(n_tasks):
        output_toconsider = outputs[:,i]
        maximum = max(output_toconsider)
        minimum = min(output_toconsider)
        task_features[i,:] = np.array([maximum, minimum])

    return (inputs,outputs,sample_intensity,task_features,offset_data,random_noise,process_values,weights_data_task)


def generate_synthetic_data_noisy(N_all, n_tasks, kernel, num_latent, num_features = 2):

	# Initialise required variables
    weights_data_task = init_list(0.0, [n_tasks])
    process_values = init_list(0.0, [num_latent])
    sample_intensity = init_list(0.0, [n_tasks])
    final_process_value = init_list(0.0, [n_tasks])
    outputs = np.ones((N_all,n_tasks))
    task_features = np.zeros((n_tasks,num_features))
    random_noise_vector = np.ones((N_all,n_tasks))

    
    # Define the tasks specific offsets. 
    offset_data = np.float32(np.array([2.0, 2.0, 2.6, 2.6]))
    np.save(os.path.join(os.path.dirname(__file__), "..", "..") + '/Data/Synthetic_Experiments/' + kernel + '/offset_data_noisy', offset_data)

	# Select some weights to generate the data
    weights_data_task[0] = np.float32(np.array([+0.1,-0.12]))
    weights_data_task[1] = np.float32(np.array([-0.1,-0.1])) 
    weights_data_task[2] = np.float32(np.array([-0.1,+0.1]))
    weights_data_task[3] = np.float32(np.array([-0.2,+0.1]))

	# Random noise added to the true parameters in order to initialise the algorithm pars
    random_noise = np.random.normal(loc=0.0, scale=1.0, size=1)

	# Initiliaze the inputs and stardardize them
    space_x = np.linspace(0.0,1.0,N_all/20)
    time_x = np.linspace(0.0,9.0,20)
    space_x,time_x = np.meshgrid(space_x,time_x)
    inputs = np.array([space_x.flatten(),time_x.flatten()]).T
    np.save(os.path.join(os.path.dirname(__file__), "..", "..") + '/Data/Synthetic_Experiments/' + kernel + '/original_inputs_synthetic', inputs)
    inputs_mean = np.transpose(np.mean(inputs))
    inputs_std = np.transpose(np.std(inputs))
    standard_inputs = (inputs - inputs_mean)/inputs_std
    inputs = standard_inputs
    
    # Kernel function
    def kernel_func(a,b,l):
        k = np.zeros((len(a),len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                k[i,j] = np.exp(-(2/l**2)*(np.sin(np.pi*np.absolute(a[i] - a[j]))**2))
        return k
            
    
    # Run the spatial and temporal kernels on the input data
    sigma1_spatial = 5 * sk.rbf_kernel(inputs[:,0][:, np.newaxis], inputs[:,0][:, np.newaxis], gamma=25)
    sigma2_spatial = 5 * sk.rbf_kernel(inputs[:,0][:, np.newaxis], inputs[:,0][:, np.newaxis], gamma=20)
    sigma1_temporal = 5 * kernel_func(inputs[:,1][:, np.newaxis], inputs[:,1][:, np.newaxis], l=1)
    sigma2_temporal = 5 * kernel_func(inputs[:,1][:, np.newaxis], inputs[:,1][:, np.newaxis], l=1.5)
    #sigma1_temporal = 5 * sk.linear_kernel(inputs[:,1][:, np.newaxis],inputs[:,1][:, np.newaxis])
    #sigma2_temporal = 5 * sk.linear_kernel(inputs[:,1][:, np.newaxis],inputs[:,1][:, np.newaxis])
    sigma1 = np.add(sigma1_spatial,sigma1_temporal)
    sigma2 = np.add(sigma2_spatial,sigma2_temporal)
    
    # Sample the true underlying GPs. 
    for i in range(num_latent):
        if i == 0:
            np.random.seed(10)
            process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma1)
            process_values[i] = np.reshape(process_values[i], (N_all,1))
        if i == 1:
            process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma2)
            process_values[i] = np.reshape(process_values[i], (N_all,1))

	# Generate the intensities as a linear combination of the latent functions with the weights specific to the task + offset specific to the task
    for i in range(n_tasks):
        weighted_sum = 0.0
        for j in range(num_latent):
            process_values_single = np.array(process_values[j])
            weights_data_task_single = np.array(weights_data_task[i])[:,np.newaxis]
            weighted_sum += weights_data_task_single[j,:]*process_values_single
            random_noise_vector[:,j] = np.random.normal(loc=0.0, scale=1.0, size=N_all)
        sample_intensity[i] = np.exp(weighted_sum + offset_data[i])
        final_process_value[i] = weighted_sum + offset_data[i]
        
    np.save(os.path.join(os.path.dirname(__file__), "..", "..") + '/Data/Synthetic_Experiments/' + kernel + '/process_values_noisy', final_process_value)
    
	# Generate the outputs by sampling from a Poisson with the constructed intensities
    for j in range(n_tasks):
        for i in range(N_all):
            sample_intensity_single = sample_intensity[j]
            outputs[i,j] = np.random.poisson(lam = sample_intensity_single[i,0]) + np.random.normal(loc=0.0, scale=4.0, size=1) + 15


	# Define some task_features used when placing a GP prior on the mixing weights
    for i in range(n_tasks):
        output_toconsider = outputs[:,i]
        maximum = max(output_toconsider)
        minimum = min(output_toconsider)
        task_features[i,:] = np.array([maximum, minimum])

    return (inputs,outputs,sample_intensity,task_features,offset_data,random_noise,process_values,weights_data_task,random_noise_vector)



def generate_missing_data_synthetic(outputs, missing_experiment):

	index1 = range(10,60)
	index1_non_missing1 = range(0,10)
	index1_non_missing2 = range(60,200)

	index2 = range(30,80)
	index2_non_missing1 = range(0,30)
	index2_non_missing2 = range(80,200)

	index3 = range(140,190)
	index3_non_missing1 = range(0,140)
	index3_non_missing2 = range(190,200)

	index4 = range(50,100)
	index4_non_missing1 = range(0,50)
	index4_non_missing2 = range(100,200)

	if missing_experiment == True:
		outputs[index1,0] = np.nan
		outputs[index2,1] = np.nan
		outputs[index3,2] = np.nan
		outputs[index4,3] = np.nan

	# Define the indeces for non missing obs data
	ytrain_non_missing_index = ~np.isnan(outputs)

	return (outputs, ytrain_non_missing_index)