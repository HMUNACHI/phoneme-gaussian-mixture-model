import numpy as np
import os
import matplotlib.pyplot as plt
from harp.print_values import *
from harp.plot_data_all_phonemes import *
from harp.plot_data import *
from harp.get_predictions import *
from harp.plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
X_full[:, 0] = f1
X_full[:, 1] = f2
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, leverage the phoneme_id array, that contains the ID of each sample of X_full

phoneme_1 = X_full[phoneme_id == 1]
phoneme_2 = X_full[phoneme_id == 2]
X_phonemes_1_2 = np.concatenate((phoneme_1, phoneme_2), axis=0)

########################################/

# as dataset X, we will use only the samples of phoneme 1 and 2
X = X_phonemes_1_2.copy()

min_f1 = int(np.min(X[:,0]))
max_f1 = int(np.max(X[:,0]))
min_f2 = int(np.min(X[:,1]))
max_f2 = int(np.max(X[:,1]))
N_f1 = max_f1 - min_f1
N_f2 = max_f2 - min_f2
print('f1 range: {}-{} | {} points'.format(min_f1, max_f1, N_f1))
print('f2 range: {}-{} | {} points'.format(min_f2, max_f2, N_f2))

#########################################
# Create a custom grid of shape N_f1 x N_f2
# The grid will span all the values of (f1, f2) pairs, between [min_f1, max_f1] on f1 axis, and between [min_f2, max_f2] on f2 axis
# Then, classify each point [i.e., each (f1, f2) pair] of that grid, to either phoneme 1, or phoneme 2, using the two trained GMMs
# Do predictions, using GMM trained on phoneme 1, on custom grid
# Do predictions, using GMM trained on phoneme 2, on custom grid
# Compare these predictions, to classify each point of the grid
# Store these prediction in a 2D numpy array named "M", of shape N_f2 x N_f1 (the first dimension is f2 so that we keep f2 in the vertical axis of the plot)
# M should contain "0.0" in the points that belong to phoneme 1 and "1.0" in the points that belong to phoneme 2
########################################/
# creating the grid
a = np.linspace(min_f1, max_f1, N_f1)
b = np.linspace(min_f2, max_f2, N_f2)
f1_m, f2_m = np.meshgrid(a, b)

# convert the mesh to an input X of the structure [[f1a,f2a],[f1b, f2b]...]
X_1 = []
for i in f1_m:
  for j in i:
    X_1.append(j)

X_2 = []
for i in f2_m:
  for j in i:
    X_2.append(j)

# final input to the pre-built get prediction function
input = np.array([[X_1[i], X_2[i]] for i in range(len(X_2))])
# get the number of samples
N = input.shape[0]
# get dimensionality of our dataset
D = input.shape[1]

# GETTING PREDICTIONS
# FOR GMM PRETRAINED IN PHONEME 1
# load the saved model for phoneme 1 with the at k
path = "/Users/mac/Dropbox/QMUL/2. Machine Learning/190031512/assgn_2/data/GMM_params_phoneme_01_k_0{}.npy".format(k)
gmm_raw = np.load(path, allow_pickle=True)
gmm = gmm_raw.tolist()

# copy the required parameters
mu = gmm['mu']
s = gmm['s']
p = gmm['p']

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z = np.zeros((N,k)) # shape Nxk

# get probabilities
Z = get_predictions(mu, s, p, input)
Z_1 = np.array(Z)

##################################################################################################################################

# FOR GMM PRETRAINED ON PHONEME 2
# load the saved model for phoneme 2
path = "/Users/mac/Dropbox/QMUL/2. Machine Learning/190031512/assgn_2/data/GMM_params_phoneme_02_k_0{}.npy".format(k)
gmm_raw = np.load(path, allow_pickle=True)
gmm = gmm_raw.tolist()

# copy the required parameters
mu = gmm['mu']
s = gmm['s']
p = gmm['p']

# Initialize array Z that will get the predictions of each Gaussian on each sample
Z = np.zeros((N,k)) # shape Nxk

# get probabilities
Z = get_predictions(mu, s, p, input)
Z_2 = np.array(Z)

# get predictions
# initialise an empty list of preds
preds = []
# loop through the Z_1 and Z_2 arrays
for i in range(Z.shape[0]):
    # get the probalities for each class
    prob_1 = sum(Z_1[i])
    prob_2 = sum(Z_2[i])

    # classify the data point under the class with the highest probability
    if prob_1 > prob_2:
        preds.append(1)
    else:
        preds.append(2)

# reshape preds into a matrix M with shape N_f1 x N_f2
preds = np.array(preds)
M = preds.reshape(N_f1, N_f2)

# transpose M to get the desired shape
M = np.transpose(M)
################################################
# Visualize predictions on custom grid

# Create a figure
#fig = plt.figure()
fig, ax = plt.subplots()

# use aspect='auto' (default is 'equal'), to force the plotted image to be square, when dimensions are unequal
plt.imshow(M, aspect='auto')

# set label of x axis
ax.set_xlabel('f1')
# set label of y axis
ax.set_ylabel('f2')

# set limits of axes
plt.xlim((0, N_f1))
plt.ylim((0, N_f2))

# set range and strings of ticks on axes
x_range = np.arange(0, N_f1, step=50)
x_strings = [str(x+min_f1) for x in x_range]
plt.xticks(x_range, x_strings)
y_range = np.arange(0, N_f2, step=200)
y_strings = [str(y+min_f2) for y in y_range]
plt.yticks(y_range, y_strings)

# set title of figure
title_string = 'Predictions on custom grid'
plt.title(title_string)

# add a colorbar
plt.colorbar()

N_samples = int(X.shape[0]/2)
plt.scatter(X[:N_samples, 0] - min_f1, X[:N_samples, 1] - min_f2, marker='.', color='red', label='Phoneme 1')
plt.scatter(X[N_samples:, 0] - min_f1, X[N_samples:, 1] - min_f2, marker='.', color='green', label='Phoneme 2')

# add legend to the subplot
plt.legend()

# save the plotted points of the chosen phoneme, as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'GMM_predictions_on_grid.png')
plt.savefig(plot_filename)

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
