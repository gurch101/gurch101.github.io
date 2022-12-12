---
title: Machine Learning Notes
date: 2021-01-15
description: Machine Learning Notes
category: notes
type: notes
---

# Machine Learning

## Coursera Machine Learning Course 1

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
https://developers.google.com/machine-learning/crash-course

https://course.fast.ai/

https://courses.dataschool.io/

https://www.coursera.org/learn/machine-learning

### Problems

use speech recognition principles to build a wav -> midi maker
