import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import glob
import csv
import os
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import math


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
##############################################################################
# Load my experiment data

def loadExperimentData():
    path = "./tuft_real_data/25June/extractedFeatures/"
    list_of_data_files = glob.glob(path+'data/*.csv')
    list_of_data_files = sorted(list_of_data_files)
    flagInitial = True
    
    for file_name in list_of_data_files:
        featureFileName = os.path.splitext(file_name)[0].split("/")[-1]
        #print featureFileName
        data = np.loadtxt(fname=file_name,delimiter=',')
        if flagInitial:
            flagInitial = False
            trainData = data
        else:
            trainData = np.vstack((trainData,data))

    #For reading the labels
    list_of_label_files = glob.glob(path+'labels/*.csv')
    list_of_label_files = sorted(list_of_label_files)
    flagInitial = True        
    for file_name in list_of_label_files:
        featureFileName = os.path.splitext(file_name)[0].split("/")[-1]
        #print featureFileName
        labels = np.loadtxt(fname=file_name,delimiter=',')
        if flagInitial:
            flagInitial = False
            trainLabel = labels
        else:
            trainLabel = np.concatenate((trainLabel,labels),axis=0)

    return trainData,trainLabel
##############################################################################
# Load and prepare data set
#
# dataset for grid search
traindata,trainlabel = loadExperimentData()
X = traindata
y = trainlabel

# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 class to has
# to make it a binary classification problem.

'''X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1'''

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.
print 1
scaler = StandardScaler()
X = scaler.fit_transform(X)
#X_2d = scaler.fit_transform(X_2d)

##############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-4, 3, 50)
gamma_range = np.logspace(-4, 3, 50)
print C_range,gamma_range
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(y, n_iter=1, test_size=0.5, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# plot the scores of the grid
# grid_scores_ contains parameter settings and scores
# We extract just the scores
scores = [100*(1 - x[1]) for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
#plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=2.0, midpoint=12.0))
counter = 0

plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), np.round(gamma_range,4), rotation=90)
plt.yticks(np.arange(len(C_range)), np.round(C_range,4))
plt.title('Misclassification %')
plt.show()