
if __name__ == '__main__':
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
    
    
    # Part 4 - Evaluating, Improving, and Tuning the ANN ##########################
        
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
                               n_jobs=-1)
    
    grid_search = grid_search.fit(X_train,Y_train)
    
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
                                   
    ###############################################################################
    
    