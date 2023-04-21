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
#########################################
X_full[:, 0] = f1
X_full[:, 1] = f2
X_full = X_full.astype(np.float32)

# initialise lists for the needed phonemes and their ground truths
X_phonemes_1_2 = []
ground_truth = []

# loop through X_full and copy the f1 and f2 of any point with phoneme id of 1 or 2
for i in range(X_full.shape[0]):
    if phoneme_id[i] == 1 or phoneme_id[i] == 2:
        X_phonemes_1_2.append([f1[i], f2[i]])
        ground_truth.append(phoneme_id[i])

# convert the lists to numpy arrays with suitable data types
X_phonemes_1_2 = np.array(X_phonemes_1_2, dtype=float)
ground_truth = np.array(ground_truth, dtype=int)

# number of GMM components
k = 3
# as dataset X, use only the samples of the chosen phoneme
X = X_phonemes_1_2.copy()
# get number of samples
N = X.shape[0]
# get dimensionality of our dataset
D = X.shape[1]

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
Z = get_predictions(mu, s, p, X)
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
Z = get_predictions(mu, s, p, X)
Z_2 = np.array(Z)

# Get the predictions
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

# calculate the accuracy
# initialise a counter to record accuracy
correct = 0
# loop through each prediction
for i in range(len(preds)):
    # if prediction id the same as the ground truth
    if preds[i] == ground_truth[i]:
        # increment the correct counter
        correct += 1

# determine the accuracy of the model
accuracy = correct / len(preds) * 100
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))