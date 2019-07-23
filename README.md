# CNN-Pipeline
Script for Convolutional Neural Network for the Shiu Lab - developed during iCER ACRES REU Program. 

## Environment Requirements
python 3.6
numpy
pandas
sklearn 0.21.2
tensorflow 2.0

Tensorflow typically requires special installation. See MSU's [HPCC instructions](https://wiki.hpcc.msu.edu/display/ITH/TensorFlow)for an example. 

## Grid Search or CNN Model 
Within this pipeline, there are two files: CNN_GridSearch.py and CNN_Model.py. The grid search file uses sklearn's RandomizedGridSearch function to search for the best combination of parameters for the model given your data. It uses Negative Mean Squared Error as a measure of accuracy for the best combination of parameters. 
The model file runs the CNN model based on your input parameters. It can also take in the results from the Grid Search (-gs gridsearch.txt) as well as a file for Feature Selection (-feature_selection featurefile.txt). This file uses Pearson's Correlation Coefficient as measure of accuracy. 

## Running the Model
To run the Grid Search:

```
python CNN_GridSearch.py -geno geno_file.csv -pheno pheno_file.csv -test test_set.txt -trait colname_from_pheno_data -save save_output.txt -model_type true
```

To run the CNN Model:
```
python CNN_Model.py -geno geno_file.csv -pheno pheno_file.csv -test test_set.txt -trait colname_from_pheno_data -save save_output.txt -model_type true 
```
