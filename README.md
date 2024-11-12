# Wine-Quality-Prediction

Finetuning Model By Doing Grid Search On Various Hyperparameters.
Below is a list of common hyperparameters that needs tuning for getting the best fit for our data. We'll try various hyperparameters settings to various splits of train/test data to find out best fit which will have almost the same accuracy for both train & test dataset or have quite less difference between accuracy.
⦁	hidden_layer_sizes - It accepts tuple of integer specifying sizes of hidden layers in multi layer perceptrons. According to size of tuple, that many perceptrons will be created per hidden layer. default=(100,)
⦁	activation - It specifies activation function for hidden layers. It accepts one of below strings as input. default=relu
⦁	'identity' - No Activation. f(x) = x
⦁	'logistic' - Logistic Sigmoid Function. f(x) = 1 / (1 + exp(-x))
⦁	'tanh' - Hyperbolic tangent function. f(x) = tanh(x)
⦁	'relu' - Rectified Linear Unit function. f(x) = max(0, x)
⦁	solver - It accepts one of below strings specifying which optimization solver to use for updating weights of neural network hidden layer perceptrons. default='adam'
⦁	'lbfgs'
⦁	'sgd'
⦁	'adam'
⦁	learning_rate_init - It specifies initial learning rate to be used. Based on value of this parameter weights of perceptrons are updated.default=0.001
⦁	learning_rate - It specifies learning rate schedule to be used for training. It accepts one of below strings as value and only applicable when solver='sgd'.
⦁	'constant' - Keeps learning rate constant through a learning process which was set in learning_rate_init.
⦁	'invscaling' - It gradually decreases learning rate. effective_learning_rate = learning_rate_init / pow(t, power_t)
⦁	'adaptive' - It keeps learning rate constant as long as loss is decreasing or score is improving. If consecutive epochs fails in decreasing loss according to tol parameter and early_stopping is on, then it divides current learning rate by 5.
⦁	batch_size - It accepts integer value specifying size of batch to use for dataset. default='auto'. The default auto batch size will set batch size to min(200, n_samples).
⦁	tol - It accepts float values specifying threshold for optimization. When training loss or score is not improved by at least tol for n_iter_no_change iterations, then optimization ends if learning_rate is constant else it decreases learning rate if learning_rate is adaptive. default=0.0001
⦁	alpha - It specifies L2 penalty coefficient to be applied to perceptrons. default=0.0001
⦁	momentum - It specifies momentum to be used for gradient descent and accepts float value between 0-1. It's applicable when solver is sgd.
⦁	early_stopping - It accepts boolean value specifying whether to stop training if training score/loss is not improving. default=False
⦁	validation_fraction - It accepts float value between 0-1 specifying amount of training data to keep aside if early_stopping is set.default=0.1
Data
For this analysis we will cover one of life’s most important topics – Wine fraud! All joking aside, wine fraud is a very real thing. Let’s see if a Neural Network in Python can help with this problem! We will use the wine data set from the UCI Machine Learning Repository. It has various chemical features of different wines, all grown in the same region in Italy, but the data is labeled by three different possible cultivars. We will try to build a model that can classify what cultivar a wine belongs to based on its chemical features using Neural Networks. You can get the data here or find other free data sets here.
First let’s import the dataset! We’ll use the names feature of Pandas to make sure that the column names associated with the data come through.
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from math import sqrt
#from sklearn.metrics import r2_score

#Import the Data
df = pd.read_csv('/Users/user/Desktop/7BUIS008W/wine.csv') 
print(df.shape)
df.describe().transpose()

#Let’s set up and split our Data and our Labels
X = df.drop('Wine',axis=1)
y = df['Wine']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

Data Pre-processing
The neural network in Python may have difficulty converging before the maximum number of iterations allowed if the data is not normalized. Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. Note that you must apply the same scaling to the test set for meaningful results. There are a lot of different methods for normalization of data, we will use the built-in StandardScaler for standardization.

#Scale the Train and Test Data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Apply Grid Search to tune your MLP
We can calculate the best parameters for the model using “GridSearchCV”. The input parameters 
for the GridSearchCV method are
1. The MLP model
2. A parameter dictionary in which we define various hidden layers, activation units, learning rates.
It trains the model and finds the best parameter.
%%time

from sklearn.model_selection import GridSearchCV

params = {'activation': ['relu', 'tanh', 'logistic', 'identity'],
          'hidden_layer_sizes': [(13,), (50,100,), (50,75,100,)],
          'solver': ['adam', 'sgd', 'lbfgs'],
          'learning_rate' : ['constant', 'adaptive', 'invscaling'],
          'max_iter': [500]
         }

mlp_classif_grid = GridSearchCV(MLPClassifier(random_state=123), param_grid=params, n_jobs=-1, cv=5, verbose=5)
mlp_classif_grid.fit(X_train,y_train)

print('Train Accuracy : %.3f'%mlp_classif_grid.best_estimator_.score(X_train, y_train))
print('Test Accuracy : %.3f'%mlp_classif_grid.best_estimator_.score(X_test, y_test))
print('Best Accuracy Through Grid Search : %.3f'%mlp_classif_grid.best_score_)
print('Best Parameters : ',mlp_classif_grid.best_params_)

 

Now that the model has been made we can fit the training data to our model, remember that this data has already been processed and scaled:
mlp = MLPClassifier(activation= 'relu', hidden_layer_sizes= (13,), learning_rate='constant', solver='lbfgs', max_iter=500)
mlp.fit(X_train,y_train)

Predictions and Evaluation

Now that we have a model it is time to use it to get predictions! We can do this simply with the predict() method off of our fitted model:
predictions = mlp.predict(X_test)
Now we can use SciKit-Learn’s built in metrics such as a classification report and confusion matrix to evaluate how well our model performed: 
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
  
print(classification_report(y_test,predictions))
 
