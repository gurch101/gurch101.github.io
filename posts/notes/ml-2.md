---
title: Advanced Learning Algorithms
date: 2021-01-15
description: Coursera ML Specialization Course 2 Notes
category: notes
type: notes
---

Neural networks

- inference/prediction
- training
- practical advice
  Decision trees

Neural networks - algorithms that try to mimic the brain using simplified mathematical models of a neuron
Used for handwriting, speech recognition, image recognition, natural language processing

A neural network has layers of neurons. Each neuron takes an input vector and outputs a single value.

Can be thought of as logistic regression where each layer learns its own features. For example, in demand prediction,
inputs could be price, shipping cost, marketing, material, output layer could engineer its own features of affordability, awareness, and perceived quality, which is then fed to another layer to output another set of features.

For image recognition, pixel intensities are fed as a vector. Each layer indentifies increasingly complex features - lines, shapes, eyes/ears, etc.

![Neural Network Layer](/images/nnlayer.png)

Forward propagation = prediction
Backward propagation = learning

Forward propagation in Python

```py
x = np.array([200,17])
w1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1,x) + b1_1
a1_1 = sigmoid(z1_1)

w1_2 = np.array([-3,4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2,x) + b1_2
a1_2 = sigmoid(z1_2)

w1_3 = np.array([5,-6])
b1_3 = np.array([1])
z1_3 = np.dot(w1_3,x) + b1_3
a1_3 = sigmoid(z1_3)

a1 = np.array([a1_1, a1_2, a1_3])

# generic implementation
def dense(a_in, W, b, g):
  """
  Computes dense layer
  Args:
    a_in (ndarray (n, )) : Data, 1 example
    W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
    b    (ndarray (j, )) : bias vector, j units
    g    activation function (e.g. sigmoid, relu..)
  Returns
    a_out (ndarray (j,))  : j units|
  """
  units = W.shape[1] # rows = number of features, columns = units
  a_out = np.zeros(units)
  for j in range(units): # if a,w,b were matrices, this loop could be replaced with Z = np.matmul(A_in,W) + b
    w = W[:,j]
    z = np.dot(w, a_in) + b[j]
    a_out[j] = g(z)
  return a_out


def sequential(x, W1, b1, W2, b2):
    a1 = dense(x,  W1, b1, sigmoid)
    a2 = dense(a1, W2, b2, sigmoid)
    return(a2)
```

### Tensorflow

Machine learning package from Google. Keras creates a simple, layer-centric interface to Tensorflow.

![Tensorflow Inference](/images/tensorflowinference.png)

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)


# layer with one neuron using linear regression
linear_layer = tf.keras.layers.Dense(units=1, activation='linear',)
# initializes weights to random values. Tensorflow requires 2d matrices
linear_layer(X_train[0].reshape(1,1))
w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")

set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])

# both produce identical values. a1 is a tensor. To convert back to a numpy array, call a1.numpy()
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
print(alin)

prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b


# Sequential is a list of nn layers
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)

# prints layers, type, shape, params
model.summary()

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)

a1 = model.predict(X_train[0].reshape(1,1))
print(a1)

# normalize the data
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# run gradient descent
model.fit(
    Xt,Yt,
    epochs=10,
)

# model.compile/model.fit(x,y) to train the model
```

### Train a 0/1 Digit Recognizer

```py
### Data Viz
m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()

### Train the model
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        tf.keras.layers.Dense(25, activation="sigmoid"),
        tf.keras.layers.Dense(15, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ], name = "my_model"
)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X,y,
    epochs=20
)

### Visualize the results

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1,400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0

    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)
plt.show()
```

### Neural Network Training

```py

# 1. Initialize the model
model = Sequential([Dense(...), Dense(...)])
# 2. Compile the model with a loss function
# this is the same as the logistic loss function
# for regression, use MeanSquaredError()
model.compile(loss=BinaryCrossentropy())
# 3. Fit the model to the data by minimizing the cost function using backpropagation
model.fit(X,y, epochs=100)
```

### Activation Functions

Activation functions allow model fit more complex data

Linear/no activation function:
z = wx + b
Sigmoid:
1/(1+e^-z)
ReLU:
max(0,Z)

![Activation Functions](/images/activationfns.png)

For output layer:
For binary classification, choose sigmoid function (probability y = 1)
For regression problems, choose linear function (y can be + or -)
For regression problems where y can only be positive, choose ReLU

For hidden layer:
ReLU is most common choice

- faster to compute
- faster to learn. flat in the left of the graph whereas sigmoid goes flat twice. Gradient descent is slower for sigmoid.

If linear function was used for all layers, model is equivalent to linear regression
If hidden layers are all linear and output layer is sigmoid, model is equivalent to logistic regression

### Multi-class Classification

classification with more than 2 labels. For example, identifying digits 0-9, identifying parts of speech.

softmax is a generalization of logistic regression for multiclass classification

![Softmax](/images/softmax.png)

![Generalized Softmax](/images/softmaxgen.png)

Smaller aj, bigger loss

![Softmax Loss](/images/softmaxloss.png)

```py
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear'),
])

# each input only has one possible category
# use from_logits=True to account for numerical roundoff error
model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))

model.fit(X,Y,epochs=100)

logit = model(X)

# gives probabilities
f_x = tf.nn.softmax(logit)

def my_softmax(z):
    ez = np.exp(z) # element-wise exponential
    sm = ez/np.sum(ez)
    return sm
```

### Multi-label Classification

classification where each input can be associated with more than one label. For example, an image that has multiple different objects in it.

Possible approaches:

- have separate neural networks for each object
- have one neural network with an output for each object. Each output unit has sigmoid activations.

### Advanced Optimization

Adam algorithm (adaptive moment estimation) adjusts alpha (learning rate) to reach minimum faster. If parameters keeps moving in same direction, increase alpha. If parameters keep oscillating, decrease alpha.

```py
# tweak the learning rate to see which performs the best
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

### Convolutional Layers

- each neuron looks at part of the previous layers inputs rather than all of them (Dense).
- faster computation
- needs less training data and is less prone to overfitting

### Debugging & Evaluation

Regression Evaluation

- split training set into training/test set. Fit params for training. Compute test error and training error. If jtrain is low and jtest is high, you have overfitting.

![Model evaluation](/images/modelevaluation.png)

Classification Evaluation

![Classification evaluation](/images/classificationevaluation.png)

Split training into training, cross-validation, and test sets. Create multiple models and minimize cost on the cross validation set. Use test set to estimate generalization error.

![Cross validation](/images/crossvalidation.png)

If Jtrain and Jcv is high, then model has high bias and underfits

If Jtrain is low and Jcv is high, then model has high variance and overfits

If Jtrain is low and Jcv is low, then model generalizes

If Jtrain is high and Jcv is higher, then model has high bias and high variance. Overfits for part of the data and underfits for part of the data.

![Cost as a function of the polynomial degree](/images/fitbydegree.png)

With regularization, high lambda leads to high bias (model keeps weights ~= 0). Low lambda leads to overfitting. Choose lambda by slowly increasing it and compare Jcv/Jtrain for each execution.

![Cost as a function of lambda](/images/fitbylambda.png)

Establish baseline error rate by comparing to human performance, competing algorithms, or guess based on experience

![Cost as a function of training set size](/images/fitbysize.png)

For high bias, Jtrain error will be higher. Adding more training data will not bring down the error rate by itself.

For high variance, Jtrain will be lower and gap between Jtrain and Jcv will be larger. Adding more train data can improve performance.

Large neural networks for medium sized data sets are low bias machines. Can continue to increase the size of the network to improve training set performance. If it does not do well on the cross-validation set, get more data. A large neural network will usually do as well or better than a smaller one so long as regularization is chosen appropriately.

![Neural Network Debugging](/images/nndebugging.png)

Neural network can be regularized in tensorflow with `kernel_regularizer=L2(0.01)` param for the layer.

Possible Improvements

If high variance:

- Get more training examples
- Try smaller sets of features
- increase lambda

If high bias:

- Try adding polynomial features
- Get additional features
- decrease lambda

### Development Process

- Choose architecture, model, data, features
- Train model
- bias, variance, and error analysis
- repeat

### Error Analysis

- Look for common traits in misclassified cross validation examples. For example, for spam classification, some categories could be pharma-related, deliberate misspellings, unusual routing, phishing, spam in embedded images
- use common categories to add features or more data that belongs in certain categories

### Adding Data

- add more data of the types where error analysis has indicated it might help. Go to unlabeled data and find more examples of a specific category
- modify an existing training example to create a new training example (data augmentation). As an example, distort or transform image/audio of a number to create new examples.
- use artificial data imputs to create new training examples from scratch (data synthesis). As an example, use different fonts/colors to create new data for an OCR task.

### Transfer Learning

Use a large model trained on something else (supervised pre-training). Keep input and hidden layers and their parameters. Replace output layer (fine tuning).

Option 1: only train output layer parameters. Works better for small data sets.

Option 2: train all parameters but initialize them using the original neural network parameters.

For images, the thought is the hidden layers can recognize edges, corners, curves, and basic shapes which could be useful for other tasks.

### Skewed Datasets

Skewed dataset - ratio to postiive/negative examples is not 1:1.

IE rare disease classification:

- 1% error on test set but in the real world, only 0.5% of patients have disease. 1% error on its own does not indicate good results.

Compute counts of true positives, true negatives, false positives, false negatives

Precision - of all examples, what fraction is actually positive? (true positives/(true positives + false positives)). High precision = high confidence of correctness.

Recall - of all examples that are positive, what fraction were accurately predicted? (true positives/(true positives + false negatives)). High recall = high confidence of not missing positives.

In an ideal case, we want high precision and high recall.

For logistic regression, we can alter precision/recall by changing the prediction threshold (ie instead of 0.5, use 0.7). Raising the threshold results in higher precision but lower recall.

![Precision vs Recall](/images/skew.png)

F1 score (harmonic mean) can be used to choose algorithm with optimal precision/recall. F1 score takes an average while emphasizing the smaller number.

F1 score = 1/(1/2(1/p + 1/r)) = 2pr/(p+r)

### Decision Trees

Works well on tabular/structured data for classification or regression tasks. Not recommended for unstructured data (images, video, audio, text). Very fast to train.

For each example, start at root node, go down path based on feature values until leaf node (output variable) is reached

Decision tree algorithm "learns" the best decision tree of all possible decision trees.

Learning Process:

1. Choose feature for root node based on the feature that maximizes purity (or minimizes impurity)
2. split examples based on node
3. DFS for each node until examples are appropriately categorized or a maximum depth is reached or the purity score is below a threshold or when the number of examples in a node is below a threshold

Entropy - measure of impurity

![entropy](/images/entropy.png)

po = 1 - p1
H(p1) = -p1log2(p1) - p0log2(p0)

```py
def compute_entropy(y):
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """

    entropy = 0

    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y)
        if p1 != 0 and p1 != 1:
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
            entropy = 0
    return entropy
```

Information gain

choose a split by maximizing the reduction in the weighted average of the entropy for each possible feature

![Information Gain](/images/infogain.png)

```py
def split_dataset(X, node_incices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):  List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    Returns:
        left_indices (list): Indices with feature value == 1
        right_indices (list): Indices with feature value == 0
    """

    left_indices = []
    right_indices = []

    ### START CODE HERE ###
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    ### END CODE HERE ###

    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):

    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed

    """
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    # You need to return the following variables correctly
    information_gain = 0

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    # Weights
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)

    #Weighted entropy
    weighted_entropy = w_left * left_entropy + w_right * right_entropy

    #Information gain
    information_gain = node_entropy - weighted_entropy

    return information_gain

def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1

    max_info_gain=0
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature

    return best_feature

tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree.
        current_depth (int):    Current depth. Parameter used during recursive call.

    """

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices)
    tree.append((current_depth, branch_name, best_feature, node_indices))

    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
```

### One-Hot Encoding

One-hot encoding is used for features that can take on multiple discrete values.

Make each category its own boolean feature. IE ear shape of pointy, oval, floppy can be three features instead.

If a categorical feature can take on k values, create k binary features (0 or 1)

### Continuous Features

Split based on whether value is <= some value. Consider multiple thresholds and use the one that produces the best information gain.

Algorithm:

- sort by value
- use midpoint values as possible threshold
- choose max info gain value

### Regression Trees

use input features to predict a continuous number. Take average of all output values in leaf node. Each node should attempt to maximize reduction in variance of output variable at each split.

![regression tree split](/images/regressiontreesplit.png)

### Tree Ensembles

Decision trees are highly sensitive to small changes of the data. Changing one training example can completely change the tree. Tree ensembles can be used to make the trees more robust. Create multiple trees, run each example through each tree and use the majority result.

Tree ensembles are constructed through sampling with replacement. Take m training examples at random and construct a tree with that data. Repeat this B times (64-128). Also known as a bagged decision tree.

For random forests:

At each node, when choosing a feature to use to split, if n features are available, pick a random subset of k < n features and allow the algorithm to only choose from that subset of features. For a large number of features, use k = sqrt(n).

For boosted trees:

Use sampling with replacement to create a new training set of size m but instead of picking for all examples with equal (1/m) probability, make it more likely to pick examples that the previous trained trees misclassify (deliberate practice analogy).

XGBoost (extreme gradient boosting)

```py
from xgboost import XGBClassifier

# Classification
model = XGBClassifier()

# Regression
model = XGBRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

```py
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv(path)
# get percentiles, mean, std of each column
data.describe()

# list all columns
data.columns

# get predictive column
y = data.Price

# get features
X = data['Rooms', 'Bathroom'. 'Landsize']

# show first bunch of rows
X.head()

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    # random_state ensures same results for each run, max_leaf_nodes is used to control tree depth
    # too deep = overfitting, too shallow = underfitting
    model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    # abs(predicted - actual) / N
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = { leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores, key=scores.get)

# now that all param decisions are made, fit to entire dataset
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

final_model.fit(X, y)
```

### Random Forests

A random forest uses many trees and makes predictions by averaging the prediction of each tree

```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

### Questions

Why do neural networks use the sigmoid function? why not just raw vaules like linear regression?

How many layers should I use? how many units per layer?
