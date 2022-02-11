---
title: Machine Learning Notes
date: 2021-01-15
description: Machine Learning Notes
category: notes
type: notes
---

# Machine Learning

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

### I/O

```py
import pandas as pd

data = pd.read_csv(path)
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
```

### Pandas

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
```


next:
https://developers.google.com/machine-learning/crash-course

https://course.fast.ai/