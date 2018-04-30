# Artificial Neural Network

# Installing Theano
#       - Opensource numerical computation library, based on numpy, can run on GPU
#       - Used primaryily to imporve apon NN in research environments

# Installing Tensorflow
#       - Opensource numerical computation library, based on numpy, can run on GPU
#       - Used primaryily to imporve upon NN in research environments

# Installing Keras
#       - Builds upon Theano and Tensorflow, similar to sklearn for machine learning 
if __name__ == "__main__":
    import multiprocessing as mp; mp.set_start_method('forkserver')

# Part 1 - Data Preprocessing #################################################
# Importing Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')                                    # Import Data file
X = dataset.iloc[:, 3:13].values                                                # Extract independent variable columns, into X
Y = dataset.iloc[:,-1].values                                                   # Extract dependent variable columns, into Y


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()                                               # LabelEncoder for country column
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])                                 # Encodes contries as orderable numbers

labelencoder_X_2 = LabelEncoder()                                               # LabelEncoder for gender column
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])                                 # Encodes genders as 0 and 1

onehotencoder = OneHotEncoder(categorical_features=[1])                         # However contries shouldn't be orderable for ANN
X = onehotencoder.fit_transform(X).toarray()                                    # onehotencoder fixes this by using multiple columns 
X = X[:, 1:]


# Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2) # Use 8000-2000 split and a random seed(for tut purposes)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()                                                         # scaler for X set
X_train = sc_X.fit_transform(X_train)                                           # Fitting to the train set first makes the test set on the same scale
X_test = sc_X.transform(X_test)                                                 # Don't fit, since its fitted to the training set
###############################################################################




# Part 2 - Making the ANN #####################################################

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential                                             # Package used to initialize the ANN
from keras.layers import Dense                                                  # Package for building the layers in the ANN


# Initialising the ANN
classifier = Sequential()                                                       # Layers defined step-by-step later

# Adding input and the first hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_shape=(11,)))     # Create first layer with 11 inputs and 6 outputs, using a rectifier activation function, and uniform small initial weights

# Adding the second hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))     # Create second layer with 6 outputs, using a rectifier activation function, and uniform small initial weights

# Adding the output layer
#       If the ouput had more than 2 output categories then the number of outputs of the last layer would have to increase
#           and the activation function would need to become a softmax function (similar to the sigmoid func but for more than 2 output categories)
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))     # Create output layer with 1 output, using a sigmoid activation function(to get probabilities of customers leaving), and uniform small initial weights

# Compiling the ANN (Applying Stochastic Gradient Decent to the ANN)
#       If output has more than 2 possibilies, use categorical_crossentropy instead
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])   # adam is a S.G.D. algorithm, and binary_crossentropy is a logarithmic loss function

# Fitting the ANN to the Training set
classifier.fit(X_train,Y_train,batch_size=10, epochs=100)

###############################################################################



# Part 3 - Making the predictions and evaluating the model ####################

# Predicting the Test set results
Y_pred = classifier.predict(X_test)                                             # test the module
Y_pred = (Y_pred > 0.5)                                                         # Y_pred is probabilities, but need 0/1 for confusion matrix
                                                                                #   so threshold at 50% probability (Can be different depending on the circumstance)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)                                            # build a confusion matrix for test results

# Compute the accuracy
ann_Accuracy = (cm[0,0] + cm[1,1])/np.matrix(cm).sum()


###############################################################################



# Part 4 - Evaluating, Improving, and Tuning the ANN ##########################

# Evaluating the ANN (Using k-fold cross validation functionality from sklearn to get a Bias-Variance measure)
import keras
from keras.models import Sequential                                             # Package used to initialize the ANN
from keras.layers import Dense                                                  # Package for building the layers in the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# function to build the ANN
def build_classifier():
    classifier = Sequential()                                                                       # Layers defined step-by-step later
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_shape=(11,))) # Create first layer with 11 inputs and 6 outputs, using a rectifier activation function, and uniform small initial weights
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))                   # Create second layer with 6 outputs, using a rectifier activation function, and uniform small initial weights
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))                # Create output layer with 1 output, using a sigmoid activation function(to get probabilities of customers leaving), and uniform small initial weights
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])            # adam is a S.G.D. algorithm, and binary_crossentropy is a logarithmic loss function
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier,X = X_train,y = Y_train,cv=10,n_jobs=1)    # Run the eval algorithm on the training set with 10 folds and using all available cpu cores (for speed)

mean = accuracies.mean()
variance = accuracies.std()


# Improving the ANN
# Dropout regularization to reduce overfitting if needed
#   - Overfitting is when the model becomes to reliant on observation unique to the training data
#   - Overfitting is observable from a large difference in accuracy between the training set and test set
#       or when there is a large variance in the k-fold cross validation accuracies
#   - Dropout Regularization algorithmicly disables indiviual neurons during the learning precess in order to increase 
#       individuality in the neurons, to prevent overfitting
import keras
from keras.models import Sequential                                             # Package used to initialize the ANN
from keras.layers import Dense                                                  # Package for building the layers in the ANN
from keras.layers import Dropout


classifier = Sequential()                                                       

classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_shape=(11,)))     
classifier.add(Dropout(p=0.1))                                                  # Add dropout to layer with 10% of the layer's neurons 

classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))     
classifier.add(Dropout(p=0.1))                                                  # Add dropout to layer with 10% of the layer's neurons 

classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))     

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
classifier.fit(X_train,Y_train,batch_size=10, epochs=100)


# Tuning the ANN
import keras
from keras.models import Sequential                                             # Package used to initialize the ANN
from keras.layers import Dense                                                  # Package for building the layers in the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# function to build the ANN
def build_classifier(optimizer,neurons_per_layer):
    classifier = Sequential()                                                                       # Layers defined step-by-step later
    classifier.add(Dense(units=neurons_per_layer,activation='relu',kernel_initializer='uniform',input_shape=(11,))) # Create first layer with 11 inputs and 6 outputs, using a rectifier activation function, and uniform small initial weights
    classifier.add(Dense(units=neurons_per_layer,activation='relu',kernel_initializer='uniform'))                   # Create second layer with 6 outputs, using a rectifier activation function, and uniform small initial weights
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))                # Create output layer with 1 output, using a sigmoid activation function(to get probabilities of customers leaving), and uniform small initial weights
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])            # adam is a S.G.D. algorithm, and binary_crossentropy is a logarithmic loss function
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

# Dictionary containing the parameters to be tooned and the values to test the tooning for    
parameters = {'batch_size': [25, 32], 
              'epochs': [100,500],
              'optimizer': ['adam','rmsprop'],
              'neurons_per_layer': [6,12,15]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs=1)

grid_search = grid_search.fit(X_train,Y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
                               
###############################################################################



# Homework Prediction #########################################################
''' Customer Stats:
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''

# My solution
customer = [600,'France','Male',40,3,60000,2,1,1,50000]

[customer[1]] = labelencoder_X_1.transform([customer[1]])                           # Encodes contries as orderable numbers

[customer[2]] = labelencoder_X_2.transform([customer[2]])                       # Encodes genders as 0 and 1

customer = onehotencoder.transform(np.matrix(customer).reshape(1,-1)).toarray()                                 # onehotencoder fixes this by using multiple columns 
customer = customer[:,1:]

customer = sc_X.transform(customer)

cust_pred = classifier.predict(customer)

# Instructor's solution
new_prediction = (classifier.predict(sc_X.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))) > 0.5)
new_prediction = (new_prediction > 0.5)

###############################################################################