---
title: Advanced Learning Algorithms
date: 2021-01-15
description: Coursera ML Specialization Course 2 Notes
category: notes
type: notes
---

# Advanced Learning Algorithms

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

### Decision Trees

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
