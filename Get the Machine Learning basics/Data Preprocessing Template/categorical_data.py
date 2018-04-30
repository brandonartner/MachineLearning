# Categorical Data Preprocessing
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


# Handle Missing Data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)                # class for completing missing data
imputer = imputer.fit(X[:, 1:3])                                                # sets imputer to correct dimensions
X[:,1:3] = imputer.transform(X[:,1:3])                                          # performs the data completion


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#   Encoding Independent Variable
labelencoder_X = LabelEncoder()                                                 # LabelEncoder for Y
X[:,0] = labelencoder_X.fit_transform(X[:,0])                                   # Encodes contries as orderable numbers
onehotencoder = OneHotEncoder(categorical_features=[0])                         # However contries shouldn't be orderable for MLA
X = onehotencoder.fit_transform(X).toarray()                                    # onehotencoder does this by using multiple columns 
                        
#   Encoding Dependent Variable
labelencoder_Y = LabelEncoder()                                                 # LabelEncoder for Y
Y = labelencoder_X.fit_transform(Y)                                             # Encodes YES/NO as orderable numbers