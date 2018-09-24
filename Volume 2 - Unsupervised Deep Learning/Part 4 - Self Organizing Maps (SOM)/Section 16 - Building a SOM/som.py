# Self Organizing Map
# SOM to detect fraud.

# Import the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Import the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Training the SOM using MiniSOM
from minisom import MiniSom

map_dims = (10,10)
num_of_iters = 100

som = MiniSom(*map_dims, X.shape[1])
som.random_weights_init(X)
som.train_random(X, num_of_iters)

# Plot the resulting map
from pylab import bone, pcolor, colorbar, plot, show

bone()							# Makes blank plot window
pcolor(som.distance_map().T)	# Adds distance map 
colorbar()						# Adds a color bar to explain what the colors in the map represent

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
	w = som.winner(x)
	plot(w[0] + 0.5, w[1] + 0.5, 
			markers[Y[i]], 
			markeredgecolor = colors[Y[i]], 
			markerfacecolor = 'None', 
			markersize = 10,
			markeredgewidth = 2) # Added 0.5 to move marker from the corner of the square in the plot to the center of the square


#show()

# Finding the frauds
threshold = 0.8

xys = np.nonzero(som.distance_map() > threshold)
potential_fraud_groups = list(zip(xys[0].tolist(), xys[1].tolist()))

mappings = som.win_map(X)
frauds = []
for ind in potential_fraud_groups:
	frauds = frauds + mappings[ind]

print(sc.inverse_transform(frauds))
frauds =  sc.inverse_transform(frauds)[:,0]
accepted_frauds = np.where(frauds == 1)
print('Potentially {} cases of fraud. Ids of all potential frauds:\n {}'.format(len(frauds) , frauds))
print('Potential fraudulant accounts that were accepted: \n {}'.format(accepted_frauds))