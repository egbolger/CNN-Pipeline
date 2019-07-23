from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import sys, os, timeit, argparse
import timeit
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing, metrics
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import class_weight


start_time = timeit.default_timer()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def main():
	########################
	### Parse Input Args ###
	########################
	parser = argparse.ArgumentParser(
		description='Grid Search to Use in Convolutional Neural Networks. \
			See README.md for more information about the pipeline usage',
		epilog='https://github.com/ShiuLab')
	
	### Input arguments ###
	## Required
	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-geno', help='Genotype dataset (example: example_binary.csv) ', required=True)
	req_group.add_argument('-pheno', help='Phenotype dataset (example: example_binary.csv) ', required=True)
	req_group.add_argument('-test', help='Test dataset', required=True)
	req_group.add_argument('-trait', help='Phenotype Trait', required=True)
	req_group.add_argument('-save', help='Name for Output File', required=True)

	## Optional Inputs
	#Convolution Layer 1
	inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
	inp_group.add_argument('-num_conv1_filters', help='Number of  filters for Convolution Layer 1', type = int, default = 8)
	inp_group.add_argument('-kernel_conv1', help='Kernel Size of Convolution Layer 1 (first number only, second defaults to 1)', type = int, default = 18)
	inp_group.add_argument('-stride_conv1', help='Stride of Convolution Layer 1 (first number only, second defaults to 1)', type = int, default = 1)
	inp_group.add_argument('-activation', help='Activation Type for Convolutional Layers and First Dense Layer', type = str, default = 'relu')
	
	#Convolution Layer 2
	inp_group.add_argument('-num_conv2_filters', help='Number of  Filters for Convolution layer 2', type = int, default = 16)
	inp_group.add_argument('-kernel_conv2', help='Kernel Size of Convolution Layer 2 (first number only, second defaults to 1)', type = int, default = 9)
	inp_group.add_argument('-stride_conv2', help='Stride of Convolution Layer 2 (first number only, second defaults to 1)', type = int, default = 1)

	#Pooling Layers
	inp_group.add_argument('-size_pooling1', help='Size of Pooling Layer 1 Filters (first number only, second defaults to 1)', type = int, default = 4)
	inp_group.add_argument('-size_pooling2', help='Size of Pooling Layer 2 Filters (first number only, second defaults to 1)', type = int, default = 4)

	#Fully Connected/Dense Layer
	inp_group.add_argument('-num_dense1_filters', help='Number of Filters in Dense layer 1', type = int, default = 32)
	inp_group.add_argument('-num_dense2_filters', help='Number of Filters in Dense layer 2', type = int, default = 1)
	inp_group.add_argument('-activation_dense2', help='Activation Type of Dense Layer 2', type = str, default = 'linear')

	#Dropout Layers
	inp_group.add_argument('-dropout_val1', help='Value for Dropout Layer 1', type = float, default = '0.2')
	inp_group.add_argument('-dropout_val2', help='Value for Dropout Layer 2', type = float, default = '0.2')
	#Learning Rate
	inp_group.add_argument('-learning_rate', help='Value for Learning Rate', type = float, default = '0.0001')
	#Clip Value
	inp_group.add_argument('-clip_value', help='Clip Value', type = float, default = '0.1')
	#Patience for Early Stopping
	inp_group.add_argument('-patience', help='Patience for Early Stopping', type = int, default = '200')
	#Min Delta
	inp_group.add_argument('-min_delta', help='Minimum Delta Value for Early Stopping', type = float, default = '0.0001')
	#Epochs
	inp_group.add_argument('-num_epochs', help='Number of Epochs', type = int, default = '6000')

	#Simple vs DeepGS
	inp_group.add_argument('-model_type', help='Simple Model or DeepGS', type = str, default = 'true')

	args = parser.parse_args()
	start_time = timeit.default_timer()

	#Create tuple values for kernel and strides sizes
	args.kernel_conv1 = tuple([args.kernel_conv1,1])
	args.stride_conv1 = tuple([args.stride_conv1,1])
	args.kernel_conv2 = tuple([args.kernel_conv2,1])
	args.stride_conv2 = tuple([args.stride_conv2,1])
	args.size_pooling1= tuple([args.size_pooling1,1])
	args.size_pooling2 = tuple([args.size_pooling2,1])

	#Renaming Parameters for easy access in Grid Search 
	num_conv1_filters = args.num_conv1_filters
	num_conv2_filters = args.num_conv2_filters
	num_dense1_filters = args.num_dense1_filters
	num_dense2_filters = args.num_dense2_filters
	kernel_conv1 = args.kernel_conv1
	kernel_conv2 = args.kernel_conv2
	activation = args.activation
	activation_dense2 = args.activation_dense2
	stride_conv1 = args.stride_conv1
	stride_conv2 = args.stride_conv2
	size_pooling1 = args.size_pooling1
	size_pooling2 = args.size_pooling2
	dropout_val1 = args.dropout_val1
	dropout_val2 = args.dropout_val2
	model_type = args.model_type

	###Read in the Data ###
	print("Loading Data....")
	geno=pd.read_csv(args.geno, index_col=0)
	print(geno.head())

	pheno=pd.read_csv(args.pheno, index_col =0)

	testSet=pd.read_fwf(args.test)

	trait = args.trait
	pheno=pheno[[trait]]

	save = args.save
	#Determines type of values in dataset: 0/1 or -1/0/1
	unique=pd.unique(geno.values.ravel())
	#unique is 0,-1,1 switch to -1,0,1
	unique = sorted(unique)
	n_unique=len(unique)
	n_instances=geno.shape[0] #rows
	n_markers=geno.shape[1] #colns


	print("Splitting Data....")
	#Splitting into Training, Testing
	with open(args.test) as test_file:
		test_instances = test_file.read().splitlines()
	try:
		test_geno = geno.loc[test_instances, :]
		train_geno = geno.drop(test_instances)
		test_pheno = pheno.loc[test_instances, :][trait]
		train_pheno = pheno.drop(test_instances)
	except:
		test_instances = [int(x) for x in test_instances]
		test_geno = geno.loc[test_instances, :]
		train_geno = geno.drop(test_instances)
		test_pheno = pheno.loc[test_instances, :][trait]
		train_pheno = pheno.drop(test_instances)
	print('Training Geno Shape')
	print(train_geno.shape)
	print('Testing Geno Shape')
	print(test_geno.shape)
	print('Training Pheno Shape')
	print(train_pheno.shape)
	print('Testing Pheno Shape')
	print(test_pheno.shape)


	print("One Hot Encoding Training Data....")
	#One Hot Encoding For Loops
	onehotlist_train_geno = [] 
	for col in train_geno.columns[:]: 
		enc_train = preprocessing.OneHotEncoder(categories = [unique], sparse=False)
		train_1d = np.array(train_geno[col].values.tolist()) 
		train_2d = np.reshape(train_1d, (train_1d.shape[0], -1)) 
		onehotmatrixcol_train_geno = enc_train.fit_transform(train_2d)
		onehotlist_train_geno.append(onehotmatrixcol_train_geno)
	onehotmatrix_train_geno = np.transpose(np.array(onehotlist_train_geno), (1,0,2))
	print("OHE Matrix Training Geno Shape Transposed")
	print(onehotmatrix_train_geno.shape)

	print("Input Shape for Grid Model...")
	#Reshaping one hot encoded geno matrics
	n_channels=1 #just marker type, not RGB data
	n_instances_train=onehotmatrix_train_geno.shape[0]
	print(onehotmatrix_train_geno.shape)
	onehotmatrix_train_geno = onehotmatrix_train_geno.reshape(n_instances_train, n_markers, n_unique, n_channels)
	print(onehotmatrix_train_geno.shape)
	
	print("Normalize Phenotype Data...")
	#Normalize phenotype values
	min_max_scaler=preprocessing.MinMaxScaler()
	train_pheno=min_max_scaler.fit_transform(train_pheno)


	print("Grid Search...")

	def create_model(learn_rate=.0001, epochs=10, num_dense1_filters=32, num_conv1_filters=32, num_conv2_filters=32, kernel_conv1 = 27, kernel_conv2 = 27, activation = 'relu'):
		
		#Simple Model Framework
		if model_type=='true':
			K.clear_session()
			model = models.Sequential()
			model.add(layers.Conv2D(num_conv1_filters, tuple([kernel_conv1,1]),stride_conv1, activation=activation,kernel_initializer= 'glorot_normal', input_shape=(n_markers,n_unique,n_channels)))
			model.add(layers.MaxPooling2D(size_pooling2, strides= (1,1)))
			model.add(layers.Flatten())
			model.add(layers.Dropout(dropout_val2))
			model.add(layers.Dense(num_dense2_filters, activation=activation))
			
			optimizer = tf.keras.optimizers.RMSprop(lr=learn_rate)
			model.compile(optimizer=optimizer, loss='mean_squared_error')

		#DeepGs Model Framework
		if model_type=='false':
			K.clear_session()
			model = models.Sequential()
			model.add(layers.Conv2D(num_conv1_filters, tuple([kernel_conv1,1]), stride_conv1, activation=activation,kernel_initializer= 'glorot_normal', input_shape=(n_markers,n_unique,n_channels)))
			model.add(layers.MaxPooling2D(size_pooling2))
			model.add(layers.Conv2D(num_conv2_filters,tuple([kernel_conv2,1]),stride_conv2, activation=activation))
			model.add(layers.MaxPooling2D(size_pooling2))
			model.add(layers.Dropout(dropout_val1))
			model.add(layers.Flatten())
			model.add(layers.Dense(num_dense1_filters, activation=activation))
			model.add(layers.Dropout(dropout_val2))
			model.add(layers.Dense(num_dense2_filters, activation=activation_dense2))

			optimizer = tf.keras.optimizers.RMSprop(lr=learn_rate)
			model.compile(optimizer=optimizer, loss='mean_squared_error')

		return model

	seed = 7
	np.random.seed(seed)
	onehotmatrix_train_geno_grid = onehotmatrix_train_geno[1:300,:,:]
	train_pheno_grid = train_pheno[1:300]

	#Build the Model for Grid Search 
	model = KerasClassifier(build_fn = create_model, batch_size=10, verbose=0)

	#Various Parameter Values
	learn_rate = [0.1, 0.0001, 0.00001, 0.000001]
	epochs = [10, 50, 100]
	num_dense1_filters=[32,16]
	num_conv1_filters=[32,16]
	num_conv2_filters=[32,16]
	kernel_conv1 = [27,9]
	kernel_conv2 = [27,9]
	activation = ["relu", "sigmoid"]

	param_grid = dict(learn_rate=learn_rate, epochs=epochs, num_dense1_filters=num_dense1_filters, num_conv1_filters=num_conv1_filters, num_conv2_filters=num_conv2_filters, kernel_conv1 = kernel_conv1, kernel_conv2 = kernel_conv2, activation= activation)	
	
	#Running Randomized Grid Search
	grid = RandomizedSearchCV(estimator=model, param_distributions = param_grid, cv=3, n_iter=80, n_jobs=1, scoring= 'neg_mean_squared_error')

	#Reformat results to a dataframe 
	grid_result = grid.fit(onehotmatrix_train_geno_grid, train_pheno_grid)
	grid_result_df = pd.DataFrame.from_dict(grid_result.cv_results_)
	print(grid_result_df)
	
	run_time = timeit.default_timer() - start_time
	
	print("Saving Results to Output File...")
	with open(args.save + "_GridSearch.txt", 'a') as out_gs:
		grid_result_df.to_csv(out_gs, header=out_gs.tell() == 0, sep='\t')

	print('\nFinished!')

if __name__ == '__main__':
    main()


