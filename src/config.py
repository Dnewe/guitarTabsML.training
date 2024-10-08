# DATA
TRAIN_PROP = 0.8 # proportion of train data compared to dev data
MULTICLASS_LABELS = True
SINGLECLASS_LABELS = False
Y_MULTICLASS_INDEXES = [0,6] # position of multi-label classification labels in data CSV
Y_SINGLECLASS_INDEXES = [6,12] # position of single-label classification labels in data CSV
X_STARTINDEX = 12 # starting position of elements in data CSV

# LEARNING HYPERPARAMETERS
ITERATIONS = 500
ALPHA = 0.1

# NEURON NETWORK
SIZE_LAYER1 = 48