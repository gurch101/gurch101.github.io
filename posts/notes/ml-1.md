---
title: Supervised Machine Learning - Regression and Classification
date: 2021-01-15
description: Coursera ML Specialization Course 1 Notes
category: notes
type: notes
---

Give computers the ability to learn without being explicitly programmed

Supervised Learning

- used most in real-world applications and has seen rapid advancements
- algorithms that learn input to output mappings given input/output examples to learn from.
- regression: predict a number amongst infinitely many possible outputs
  - housing prices
- classification: predict a finite number of categories
  - tumor classification

Unsupervised Learning

- given inputs but not outputs - goal is to find something interesting in unlabeled data
- recommender systems/reinforcement learning
- clustering: group together similar data
  - similar news articles
  - DNA microarray clustering
  - customer/market segmentation
- anomaly detection: find unusual data points
- dimensionality reduction: compress data using fewer numbers

### Linear Regression

- Plot data on x/y plot. Add line of best fit, use formula for line to predict other inputs.
- Also useful to put data in a table where inputs/outputs are separate columns.
- training set: input and output dataset used to train the model
- x = input variable/feature
- y = output variable/target
- m = number of training examples
- (x, y) = single training example
- w/b are model parameters (coefficients/weights)

- given a training set, and a learning algorithm, generate a function that can predict y given x
- fwb(x) = wx + b

### Sum of Squared Error

- goal of cost function is to find w,b such that y prediction is close to y target for all (x,y)
- squared error cost function: J(w,b) = 1 / 2m \* sum((ypred - ytarget) ^ 2)
- divide by 2m so that we dont get bigger errors for a larger training set. Using 2 makes the gradient descent derivative cleaner.
- square the error so that the sum doesnt go to zero when -ve/+ve errors are added together
- goal is to minimize J(w,b)
- a line in the model graph is a point on the cost graph

Simple example where b = 0

![Sum of Squared Errors Graph](/images/sse.png)

w and b - sum of squared error cost function for linear regression will always be bowl shaped
![Contour Graph](/images/contour.png)

### Gradient Descent

- Used to find w, b that minimizes J(w,b). Gradient descent can be used to minimize any function. Finds local minima.
- Algorithm:

  - start with some w,b (for linear regression, use 0,0)
  - keep changing w,b to reduce J(w,b)

    - w = w - a \* d/dwJ(w,b)
    - b = b - a \* d/dbJ(w,b)
    - a = learning rate (between 0 and 1). Controls the size of the "step" taken when changing a parameter

  - repeat until w/b no longer change much

![Gradient Descent](/images/gradientdescent.png)

![Formula](/images/gradientdescentformula.png)

If the learning rate is too large, it might never converge and may get further from the minimum. If too small, it will take very long.

Even when using a fixed learning rate, each update step is smaller since the slope is smaller as we approach the local minima.

For sum of squared errors, gradient descent will always find the global minimum.

Batch gradient descent looks at all training examples.

To make sure gradient descent is converging, plot J (y axis) vs # iterations (x axis). This is known as the learning curve. If the cost sometimes goes up, the learning rate might be too large.

![Learning Curve](/images/learningcurve.png)

Alternative: if cost decreases by <= epsilon in one iteration, then declare convergence

```py
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w = 200
b = 100
tmp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title('Housing Prices')
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()

def compute_model_output(x, w, b):
    # m is the number of training examples
    m = x.shape[0] # alt is len(x_train)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

def compute_cost(x, y, w, b):
  m = x.shape[0]
  cost_sum = 0
  for i in range(m):
    f_wb = w * x[i] + b
    cost = (f_wb - y[i]) ** 2
    cost_sum = cost_sum + cost
  return (1 / (2 * m)) * cost_sum

def compute_gradient(x, y, w, b):
  m = x.shape[0]
  dj_dw = 0
  dj_db = 0
  for i in range(m):
    f_wb = w * x[i] + b
    dj_dw_i = (f_wb - y[i]) * x[i]
    dj_db_i = f_wb - y[i]
    dj_db += dj_db_i
    dj_dw += dj_dw_i
  dj_dw = dj_dw/m
  dj_db = dj_db/m
  return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
  w = copy.deepcopy(w_in)
  J_history = []
  p_history = []
  b = b_in
  w = w_in
  for i in range(num_iters):
    dj_dw, dj_db = gradient_function(x, y, w, b)
    b = b - alpha * dj_db
    w = w - alpha * dj_dw

    if i < 100000:
      J_history.append(cost_function(x,y,w,b))
      p_history.append([w,b])
  return w, b, J_history, p_history


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
```

##### Multiple Linear Regression

Support multiple features

xj = jth feature

n = number of features

x^i = features of the ith training example

X^ij = value of j feature in ith training example

model:
fwb(x) = w1x1 + w2x2 + ... + wnxn + b

store weights in a row vector
store x in a row vector
b is a single number

fwb(x) = w dotproduct x + b

dotproduct is elementwise sum of products

```py
w = np.array([1.0,2.5,-3.3])
b = 4
x = np.array([10,20,30])
# computer does product in parallel rather than a serial for loop
f = np.dot(w,x) + b
```

##### Linear Regression by Hand

```py
import copy, math
import numpy as np

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict_single_loop(x, w, b):
  return np.dot(x, w) + b


def compute_cost(X, y, w, b):
  m = X.shape[0]
  cost = 0.0
  for i in range(m):
    f_wb_i = np.dot(X[i], w) + b
    cost = cost + (f_wb_i - y[i])**2
  return cost / (2*m)

def compute_gradient(X, y, w, b):
  m,n = X.shape
  dj_dw = np.zeros((n,))
  dj_db = 0

  for i in range(m):
    err = (np.dot(X[i], w) + b) - y[i]
    for j in range(n):
      dj_dw[j] = dj_dw[j] + err * X[i,j]
    dj_db = dj_db + err

  return dj_dw / m, dj_db / m

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history #return final w,b and J history for graphing

def zscore_normalize_features(X):
  # avg of each column
  mu = np.mean(X, axis=0)
  # stddev of each column
  sigma = np.std(X, axis=0)
  X_norm = (X - mu) / sigma
  return (X_norm, mu, sigma)

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 1.0e-1
X_norm, mu, sigma = zscore_normalize_Features(X_train)
# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_norm, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations)

# to predict new input, normalize the features
x_house = np.array([1200,3,1,40])
x_house_norm = (x_house - X_mu) / X_sigma
x_house_predict = np.dor(x_house_norm, w_final) + b_final
```

Normal equation can be used as an alternative to gradient descent for linear regression without iterations. It doesn't generalize to other learning algorithms and it can be slow for a large number of features.

### Feature Scaling

some inputs can take a large range of values, others a small range of values (ex sqft vs # of bedrooms). Large inputs will likely have small weights, small values will have larger weights. The contour plot of the cost function will be asymmetric signifying small changes in one dimension can have a large impact - takes longer to find a local minima. Scaling features to operate on a similar scale produces a contour plot that is more circular -> a more direct path to the minima can be found.

can be scaled by input feature/max(feature across all rows)

alt: mean normalization - (input feature - mean)/(max - min)

alt: z-score normalization - (input feature - mean)/stddev. After z-score normalization, all features will have a mean of 0 and a standard deviation of 1.

aim to get features roughly between -1 to 1. Only need to rescale features that are too large or too small.

Feature scaling will improve the performance of gradient descent since it makes a more direct path to a minimum.

### Feature Engineering

Use intuition to design new features by transforming or combining original features. For example, for housing prices dataset that has lot frontage and depth as separate features, you can create a new feature to represent area.

### Polynomial Regression

If data is nonlinear, polynomial regression can be used to get curved lines. Simply add features that are raised to a power and run linear regression. Gradient descent naturally 'picks' the 'correct' feature by emphasizing its associated weight. The best features will be linear relative to the target (plot X^n against Y to see if linear line is generated).

##### Linear Regression with Scikit-Learn

```py
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScalar

scaler = StandardScaler()
# z-score normalization
X_norm = scaler.fit_transform(X_train)
# selects a sparse subset of features most relevant to the target variable
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
# sgdr._n_iter_, sgdr.t_ = number of completed iteration, number of weight updates
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm # y_pred = y_pred_sgd

# plot predictions and targets vs original features
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# alt:
# show the line of best fit in single input variable
plt.plot(x_train, y_pred, c = 'b')
# show a scatter plot of the raw data
plt.scatter(x_train, y_train, marker='x', c='r')
```

##### Categorical data

One-hot encoding can be used to represent categorical variables as numerical data.

```py
pd.get_dummies(data=df, drop_first=True)
```

```
Color
red
green
blue
```

is converted to:

```
red green blue
1   0     0
0   1     0
0   0     1
```

A label encoder can be used to replace categories with a number from 0 to num categories

```py
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

cat_col = []
for col in train_data.select_dtypes('object'):
  cat_col.append(col)
train_data[cat_col] = encoder.fit_transform(cat_col)
```

```
Color
red
green
blue
```

is converted to:

```
Color
0
1
2
```

### Kaggle

```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import seaborn as sns

# don't replace NA with NaN
train_data = pd.read_csv('/path/to/csv', na_filter=False)
# remove actual NaN rows
train_data.dropna(inplace=True)
# one-hot encoding
train_data = pd.get_dummies(data=df, drop_first=True)

X = train_data.drop(['SalePrice', 'Id'], axis=1)
y = train_data['SalePrice']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,train_size=0.7)
XG = XGBRegressor()
XG.fit(x_train,y_train)
score = XG.score(x_test,y_test)

sgdr = SGDRegressor()
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X_norm,Y,test_size=0.3,train_size=0.7)
sgdr.fit(x_train, y_train)
score = sgdr.score(x_test,y_test)

print(mean_absolute_error(y_test, sgdr.predict(x_test)))

# get the highest correlating features
corr_m = x_train.corr()
fig2, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(corr_m, vmax=0.8, square=True,cmap="coolwarm",annot=True)
plt.show()
highest_corr_features = corr_m.index[abs(corr_m["SalePrice"]) > 0.5]
```

### Classification

output variable can take on a small number of possible values

logistic regression is used to solve classification problems

examples:
is email spam?
is transaction fraudulent?
is tumor malignant?

Linear regression is impacted by outliers which can shift the decision boundary/threshold

![linear regression for classification](/images/regression-for-classification.png)

Logistic regression is used to classify data into 0 and 1. Can use sigmoid function (1 / (1 + e^-z)). (s curve the has y intercept at 0.5, between 0 and 1). Set z = w dotproduct x + b. Result can be thought of as the probability that class is 1.

```py
def sigmoid(z):
  g = 1/(1 + np.exp(-z))
  return g
```

If a threshold of 0.5 is used, model predicts 1 when w dotproduct x + b >= 0. For non-linear decision boundaries, use same principles of polynomial regression.

Sum of squared errors is not a good cost function for logistic regression because its not a convex - gradient descent will fall into a local minimum, not the global minimum.

Instead, use logistic loss function which is convex and can reach a global minimum:

![Logistic Loss](/images/logistic-loss.png)
![Logicistic Loss Negative Case](/images/logistic-loss-negative.png)
![Simplified Loss](/images/simplified-loss.png)

Logistic Cost function:
![Logistic Cost](/images/logistic-cost.png)

Gradient Descent
![Gradient Descent for Logistic Regression](/images/gradient-descent-logistic-regression.png)

```py
def compute_gradient_logistic(X, y, w, b):
  m,n = X.shape
  dj_dw = np.zeros((n,))
  dj_db = 0.
  for i in range(m):
    f_wb_i = sigmoid(np.dot(X[i],w) + b)
    err_i = f_wb_i - y[i]
    for j in range(n):
      dj_dw[j] = dj_dw[j] + err_i * X[i,j]
    dj_db = dj_db + err_i
  dj_dw = dj_dw/m
  dj_db = dj_db/m
  return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history         #return final w,b and J history for graphing

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)
y_pred = lr_model.predict(X)
print("Accuracy on training set:", lr_model.score(X, y))
```

### Overfitting

underfitting = model does not fit the training set well. High bias.
generalization = fits training set and brand new examples well.
overfitting = model fits the training set extremely well but will not generalize to new examples. High variance.

Overfitting can be addressed by:

1. collecting more training examples.
2. Use fewer features
3. Regularization - reduce the size of weights. Makes the line smoother.

![Regularization](/images/regularization.png)

![Regularized Gradient Descent](/images/regularizedgradientdescent.png)

above minimizes the mean squared error while trying to keep wj small.

If lambda is 0, model overfits. If lambda is very large, model underfits (f(x) = b).

```py
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """

    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot
        cost = cost + (f_wb_i - y[i])**2                               #scalar
    cost = cost / (2 * m)                                              #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar

    total_cost = cost + reg_cost                                       #scalar
    return total_cost

def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar

    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar

    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar

def compute_gradient_linear_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw

def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw
```

# Numpy

```py
import numpy as np

# vector creation
a = np.zeros(4)
# returns vector of 4 elements from 0-1
a = np.random.random_sample(4)
returns vector of 0,1,2,3
a = np.arange(4.)
a = np.random.rand(4)
a = np.array([1,2,3,4])
# one row of five elements
a = np.zeros((1,5))
# vectors have a shape property
a.shape
# slicing is done with start:stop:step
a[2:7:1]
a[3:]
a[:]
a[:, 2:7:1]
# operations
np.sum(a)
np.mean(a)
a**2
# element-wise sum/product
a + a
a * 5
```

# Pandas

### I/O

```py
import pandas as pd

data = pd.read_csv(path)
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```

### Common Operations

```py
import pandas as pd

# table with two rows, columns YES and NO
# index is the row label, if not provided, uses 0...n-1 by default
pd.DataFrame({'Yes': [50,21], 'No': [131,2]}, index=['Product 1', 'Product 2'])

# sequence/list of values, can be thought of as single column
pd.Series([1,2], index=['2015 sales', '2016 sales'], name='Product Sales')

data = pd.read_csv('path/to/csv.csv', index_col=<0 to n-1>)

# (<rows>, <columns>)
data.shape

data.to_csv('path/to/write.csv')


# column access
data['column name']

# row access
data['column name'][0]

# first three rows of first column
data.iloc[:3, 0]

# first three rows of column name
data.loc[:3, 'column name']

# all rows for n columns
data.loc[:, ['col1', 'col2']]

# change the index
data.set_index('col1')

# querying
data.loc[data['col name'] == 'foobar' & (data['colX'].isin([val1, val2])) & (data['colN'].notnull()) & (data['col2'] >= 10 | data['col3'] < 1)]

# assignment
data['col name'] = 'FOO'

# if numerical, shows percentiles
# if string, shows uniques, top, count
data['col name'].describe()

# frequency table
data['col name'].value_counts()

# alt for above
data.groupby('col name')['col name'].count()

# returns all unique values in col
data['col name'].unique()

# doesnt mutate existing data frame
data['col name'].map(lambda v: v + 100)

# equiv to above
data['col name'] + 100

# the index of the row with the max col name
data['col name'].idxmax()


# calls func_name with row
# if axis='index', calls func_name for each column
# returns new dataframe, doesn't mutate existing
data.apply(func_name, axis='columns')

# min price for each point category
reviews.groupby('points').price.min()

# best wine for country/province
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

reviews.groupby(['country']).price.agg([len, min, max])


# make multi-index back to single index
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len', ascending=False)
countries_reviewed.sort_values(by=['country', 'len'])

best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()

# strings are of type object
reviews['col name'].dtype
# data frame of column -> type
reviews.dtypes
# cast column to another type
reviews['col name'].astype('float64')
# get all reviews with null column
reviews[pd.isnull(reviews['country'])]
reviews.country.fillna('some val')
reviews.country.replace('from', 'to')

reviews.rename(columns={'country': 'somethingelse'})


# put labels on each axis
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

# combine rows of df1 and df2 in new df
pd.concat([df1, df2])

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
```

next:
https://www.coursera.org/projects/used-car-price-prediction-using-machine-learning-models

https://developers.google.com/machine-learning/crash-course

https://course.fast.ai/

https://courses.dataschool.io/

https://www.coursera.org/learn/machine-learning

https://www.amazon.ca/Grokking-Machine-Learning-Luis-Serrano/dp/1617295914

https://www.amazon.ca/Grokking-Artificial-Intelligence-Algorithms-Hurbans/dp/161729618X/ref=pd_lpo_2?pd_rd_w=2IkSF&content-id=amzn1.sym.bc8b374c-8130-4c45-bf24-4fcc0d96f4d6&pf_rd_p=bc8b374c-8130-4c45-bf24-4fcc0d96f4d6&pf_rd_r=50QQ4KNVZFECYAE6ZK4Y&pd_rd_wg=4gcj6&pd_rd_r=df9b5ab5-7424-43b0-a89b-1429044164f8&pd_rd_i=161729618X&psc=1

https://www.amazon.ca/Grokking-Deep-Learning-Andrew-Trask/dp/1617293709/ref=pd_bxgy_sccl_1/139-6089511-3105238?pd_rd_w=Nenof&content-id=amzn1.sym.17b2b149-58e2-4824-ba79-851c5f351fdc&pf_rd_p=17b2b149-58e2-4824-ba79-851c5f351fdc&pf_rd_r=50QQ4KNVZFECYAE6ZK4Y&pd_rd_wg=HJRUU&pd_rd_r=85270beb-f7ff-4720-b6b0-e8165b411269&pd_rd_i=1617293709&psc=1

Learn about scikit learn GridSearchCV

https://e2eml.school/blog.html

https://cds.nyu.edu/deep-learning/

https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/

https://github.com/oxford-cs-deepnlp-2017

### Problems

use speech recognition principles to build a wav -> midi maker
build a mike tysons punch out solver
make a deepfake to put your face throughout history
crawl every Dataphile spec and make it searchable

### Questions

which features should I use? Maybe use GridSearchCV?
how do I know if I need to engineer features?
what do you do with dates?
how do you with deal with NaNs?
how do you do non-binary classification?
