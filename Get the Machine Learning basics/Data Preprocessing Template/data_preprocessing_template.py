# Data Preprocessing 
#   Template of usual series of steps needed before building a Machine Learning Algorithm
#       1. Importing the Dataset
#       2. Handle Missing Data (Not always necessary)
#       3. Encoding Categorical Data (Not always necessary)
#       4. Splitting the Dataset into the Training set and Test set
#       5. Feature Scaling
#

# Importing Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values


# Splitting the Dataset into the Training set and Test set
    # from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split                            # cross_validation depreciated, use model_selection

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0) # Use 8-2 split and a random seed(for tut purposes)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()                                                         # 
X_train = sc_X.fit_transform(X_train)                                           # Fitting to the train set first makes the test set on the same scale
X_test = sc_X.transform(X_test)                                                 # Don't fit, since its fitted to the training set
