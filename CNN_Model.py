from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import timeit
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import sys, os, timeit, argparse
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
		description='Predicting Traits Using Convolutional Neural Networks. \
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

	##Optional Inputs

	inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
	inp_group.add_argument('-gs', help='Name for GS File', required=False)
	inp_group.add_argument('-model', help='Model Type (CNN)', required=False, type = str, default = "CNN")
	#Feature Selection
	inp_group.add_argument('-feature_selection', help="Feature selection dataset", default = "")

	#CNN Parameters
	#Convolution Layer 1
	inp_group.add_argument('-num_conv1_filters', help='Number of  Filters for Convolution Layer 1', type = int, default = 8)
	inp_group.add_argument('-kernel_conv1', help='Kernel Size of Convolution Layer 1 (first number only, second defaults to 1)', type = int, default = 18)
	inp_group.add_argument('-stride_conv1', help='Stride of Convolution Layer 1 (first number only, second defaults to 1)', type = int, default = 1)
	inp_group.add_argument('-activation', help='Activation Type for Convolutional Layers and first Dense Layer if more than one', type = str, default = 'relu')
	inp_group.add_argument('-activation_conv1', help='Activation Type of Convolution Layer 1', type = str, default = 'relu')
	
	#Convolution Layer 2
	inp_group.add_argument('-num_conv2_filters', help='Number of  Filters for Convolution Layer 2', type = int, default = 16)
	inp_group.add_argument('-kernel_conv2', help='Kernel Size of Convolution Layer 2 (first number only, second defaults to 1)', type = int, default = 9)
	inp_group.add_argument('-stride_conv2', help='Stride of Convolution Layer 2 (first number only, second defaults to 1)', type = int, default = 1)
	inp_group.add_argument('-activation_conv2', help='Activation Type of Convolution Layer 2', type = str, default = 'relu')

	#Pooling Layers
	inp_group.add_argument('-size_pooling1', help='Size of pooling Layer 1 Filters (first number only, second defaults to 1)', type = int, default = 4)
	inp_group.add_argument('-size_pooling2', help='Size of pooling Layer 2 Filters (first number only, second defaults to 1)', type = int, default = 4)

	#Fully Connected/Dense Layer
	inp_group.add_argument('-num_dense1_filters', help='Number of Filters in Dense Layer 1', type = int, default = 32)
	inp_group.add_argument('-activation_dense1', help='Activation Type of Dense Layer 1', type = str, default = 'relu')
	inp_group.add_argument('-num_dense2_filters', help='Number of Filters in Dense Layer 2', type = int, default = 1)
	inp_group.add_argument('-activation_dense2', help='Activation Type of Dense Layer 2', type = str, default = 'linear')

	#Dropout Layers
	inp_group.add_argument('-dropout_val1', help='Value for Dropout Layer 1', type = float, default = '0.2')
	inp_group.add_argument('-dropout_val2', help='Value for Dropout Layer 2', type = float, default = '0.2')
	#Learning Rate
	inp_group.add_argument('-learn_rate', help='Value for Learning Rate', type = float, default = '0.0001')
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
	args.kernel_conv1 = tuple([args.kernel_conv1,1])
	args.stride_conv1 = tuple([args.stride_conv1,1])
	args.kernel_conv2 = tuple([args.kernel_conv2,1])
	args.stride_conv2 = tuple([args.stride_conv2,1])
	args.size_pooling1= tuple([args.size_pooling1,1])
	args.size_pooling2 = tuple([args.size_pooling2,1])

	
	###Read in the Data ###
	print("Loading Data....")
	#Read in genotype file, if feature file given only keep those
	if os.path.isfile(args.geno):
		geno = pd.read_csv(args.geno,index_col = 0)
		print(geno.head())
		if args.feature_selection != '':
			with open(args.feature_selection) as f:
				features = f.read().strip().splitlines()
			geno = geno.loc[:,features]
      
	elif os.path.isdir(args.geno):
		geno = pd.read_csv(args.geno, index_col=0)

	feat_list = list(geno.columns)
	print("\n\nTotal number of instances: %s" % (str(geno.shape[0])))
	print("\nNumber of features used: %s" % (str(geno.shape[1])))


	pheno=pd.read_csv(args.pheno, index_col =0)
	print("Phenotype Data")
	print(pheno.head())

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

	print("Loading Grid Search....")
	#Pull in Best Parameters from Grid Search and Save to use in Model
	gs_res = pd.read_csv(args.gs, sep='\t')
	gs_res.fillna(0, inplace=True) 
	gs_ave = gs_res.groupby(['param_num_dense1_filters', 'param_num_conv2_filters', 'param_num_conv1_filters', 'param_learn_rate', 'param_kernel_conv2', 'param_kernel_conv1', 'param_epochs', 'param_activation']).agg({'mean_test_score': 'median', 'std_test_score': 'median'}).reset_index()
	gs_ave.columns = ['param_num_dense1_filters', 'param_num_conv2_filters', 'param_num_conv1_filters', 'param_learn_rate', 'param_kernel_conv2', 'param_kernel_conv1', 'param_epochs', 'param_activation', 'mean_test_score_med', 'std_test_score_med']
	results_sorted = gs_ave.sort_values(by='mean_test_score_med', ascending=False)
	print('\nSnapshot of grid search results:')
	print(results_sorted.head())

	num_dense1_filters=results_sorted['param_num_dense1_filters'].iloc[0]
	num_conv1_filters=results_sorted['param_num_conv1_filters'].iloc[0]
	num_conv2_filters=results_sorted['param_num_conv2_filters'].iloc[0]
	learn_rate=float(results_sorted['param_learn_rate'].iloc[0])
	kernel_conv1=results_sorted['param_kernel_conv1'].iloc[0]
	kernel_conv2=results_sorted['param_kernel_conv2'].iloc[0]
	num_epochs=results_sorted['param_epochs'].iloc[0]
	activation=results_sorted['param_activation'].iloc[0]

	print("Splitting Data....")
	#Splitting into Training, Validation, Testing
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
	print('Testing Pheno Save')
	print(test_pheno.shape)

	#Splitting into Training and Validating
	train_geno, val_geno, train_pheno, val_pheno = train_test_split(train_geno, train_pheno, test_size=0.125)
	print('Validating Geno Shape')
	print(val_geno.shape)
	print('Validating Pheno Shape')
	print(val_pheno.shape)

	print("One Hot Encoding Data....")
	#One Hot Encoding For Loops
		#One Hot Encoding Training Geno 
	onehotlist_train_geno = [] 
	for col in train_geno.columns[:]: 
		enc_train = preprocessing.OneHotEncoder(categories = [unique], sparse=False)
		train_1d = np.array(train_geno[col].values.tolist()) 
		train_2d = np.reshape(train_1d, (train_1d.shape[0], -1)) 
		onehotmatrixcol_train_geno = enc_train.fit_transform(train_2d)
		onehotlist_train_geno.append(onehotmatrixcol_train_geno)
	onehotmatrix_train_geno = np.transpose(np.array(onehotlist_train_geno), (1,0,2))
	print("OHE Matrix Training Geno Shape")
	print(onehotmatrix_train_geno.shape)

		#One Hot Encoding Validating Geno 
	onehotlist_val_geno = [] 
	for col in val_geno.columns[:]:
		enc_val = preprocessing.OneHotEncoder(categories = [unique], sparse=False)
		val_1d = np.array(val_geno[col].values.tolist()) 
		val_2d = np.reshape(val_1d, (val_1d.shape[0], -1)) 
		onehotmatrixcol_val_geno = enc_val.fit_transform(val_2d) 
		onehotlist_val_geno.append(onehotmatrixcol_val_geno) 
	onehotmatrix_val_geno = np.transpose(np.array(onehotlist_val_geno), (1,0,2))
	print("OHE Matrix Validation Geno Shape")
	print(onehotmatrix_val_geno.shape)


		#One Hot Encoding Testing Geno 
	onehotlist_test_geno = [] 
	for col in test_geno.columns[:]: 
		enc_test = preprocessing.OneHotEncoder(categories = [unique], sparse=False)
		test_1d = np.array(test_geno[col].values.tolist()) 
		test_2d = np.reshape(test_1d, (test_1d.shape[0], -1)) 
		onehotmatrixcol_test_geno = enc_test.fit_transform(test_2d) 
		onehotlist_test_geno.append(onehotmatrixcol_test_geno) 
	onehotmatrix_test_geno = np.transpose(np.array(onehotlist_test_geno), (1,0,2))
	print("OHE Matrix Testing Geno Shape")
	print(onehotmatrix_test_geno.shape)	
	
	print("Adjusting Input Shape for Model...")
	#Reshaping one hot encoded geno matrices to CNN input format
	n_channels=1 #just marker type, not RGB data
	n_instances_train=onehotmatrix_train_geno.shape[0]
	n_instances_test=onehotmatrix_test_geno.shape[0]
	n_instances_val=onehotmatrix_val_geno.shape[0]
	onehotmatrix_train_geno = onehotmatrix_train_geno.reshape(n_instances_train, n_markers, n_unique, n_channels)
	onehotmatrix_test_geno = onehotmatrix_test_geno.reshape(n_instances_test, n_markers, n_unique, n_channels)
	onehotmatrix_val_geno = onehotmatrix_val_geno.reshape(n_instances_val, n_markers, n_unique, n_channels)
	print("OHE Matrix Training Geno Shape")
	print(onehotmatrix_train_geno.shape)
	print("OHE Matrix Testing Geno Shape")
	print(onehotmatrix_test_geno.shape)
	print("OHE Matrix Validating Geno Shape")
	print(onehotmatrix_val_geno.shape)
	
	#Reshape the testing pheno type into 2d array
	test_pheno = np.array(test_pheno.values.tolist()) 
	test_pheno = np.reshape(test_pheno, (test_pheno.shape[0], -1))

	print("Normalize Phenotype Data....")
	#Normalize phenotype values between 0 and 1
	min_max_scaler=preprocessing.MinMaxScaler()
	train_pheno=min_max_scaler.fit_transform(train_pheno)
	min_max_scaler=preprocessing.MinMaxScaler()
	test_pheno=min_max_scaler.fit_transform(test_pheno)
	min_max_scaler=preprocessing.MinMaxScaler()
	val_pheno=min_max_scaler.fit_transform(val_pheno)


	#Creating CNN model using Tensorflow
	print("Creating CNN...")

	#Simple Model Framework
	if args.model_type=='true':
		model = models.Sequential()
		model.add(layers.Conv2D(args.num_conv1_filters, args.kernel_conv1, args.stride_conv1, activation=args.activation_conv1,kernel_initializer= 'glorot_normal', input_shape=(n_markers,n_unique,n_channels)))
		model.add(layers.MaxPooling2D(args.size_pooling2, strides= (1,1)))
		model.add(layers.Flatten())
		model.add(layers.Dropout(args.dropout_val2))
		model.add(layers.Dense(args.num_dense2_filters, activation=args.activation_dense2))

	#DeepGS Model Framework
	if args.model_type=='false':
		model = models.Sequential()
		model.add(layers.Conv2D(args.num_conv1_filters, args.kernel_conv1, args.stride_conv1, activation=args.activation_conv1,kernel_initializer= 'glorot_normal', input_shape=(n_markers,n_unique,n_channels)))
		model.add(layers.MaxPooling2D(args.size_pooling2))
		model.add(layers.Conv2D(args.num_conv2_filters, args.kernel_conv2, args.stride_conv2, activation=args.activation_conv2))
		model.add(layers.MaxPooling2D(args.size_pooling2))
		model.add(layers.Dropout(args.dropout_val1))
		model.add(layers.Flatten())
		model.add(layers.Dense(args.num_dense1_filters, activation=args.activation_dense1))
		model.add(layers.Dropout(args.dropout_val2))
		model.add(layers.Dense(args.num_dense2_filters, activation=args.activation_dense2))

	model.summary()

	#Compile and train the model 
	print("Compiling and Training Model")

	#Customized metric: Pearson's Correlation Coefficient
	def tfp_pearson(y_true, y_pred):
		return tfp.stats.correlation(y_pred, y_true, event_axis = None)
	optimizer = tf.keras.optimizers.RMSprop(lr=args.learn_rate, clipvalue=args.clip_value)
	model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tfp_pearson])

	#Early Stop training after reaching certain threshold
	earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', min_delta = args.min_delta, patience=args.patience,verbose=1)
	model.fit(onehotmatrix_train_geno, train_pheno, batch_size = 30, epochs=args.num_epochs, verbose=2, callbacks=[earlystop_callback], validation_data=(onehotmatrix_val_geno,val_pheno))
	print(onehotmatrix_train_geno.shape)
	print(train_pheno.shape)

	print(model.predict(onehotmatrix_val_geno))
	#Evaluate the model-returns loss value & metric values

	print("Evaluating Model...")
	test_loss, test_acc = model.evaluate(onehotmatrix_test_geno,test_pheno)

	#Print results related to metrics in model.fit
	print("Printing Results...")
	print(test_acc)

	run_time = timeit.default_timer() - start_time
	print(run_time)

	print("Saving Results to Output File...")
	if not os.path.isfile(save):
		output1 = open(save, 'w')
		output1.write('Genotype\tModel\tTrait\tTesting\tModel_Type\tLearning_Rate\tMin_Delta\tPatience\tConv1_Filters\tKernel\tStride\tSize_Pooling\tDropout\tTest_Loss\tTest_Acc\tRunTime\n')
		output1.close()


	output2=open(args.save, "a")
	output2.write('%s\t%s\t%s\t%s\t%s\t%f\t%f\t%f\t%s\t%s\t%s\t%s\t%f\t%f\t%f\t%f\t\n' % (args.geno, args.model,args.trait, args.test, args.model_type, args.learn_rate, args.min_delta, args.patience, args.num_conv1_filters, args.kernel_conv1, args.stride_conv1, args.size_pooling1, args.dropout_val1, test_loss, test_acc, run_time))
	output2.close()
	print('\nFinished!')

if __name__ == '__main__':
    main()





	