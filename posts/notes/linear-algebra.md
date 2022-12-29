---
title: Linear Algebra Notes
date: 2022-12-28
description: Notes on Linear Algebra
category: notes
type: notes
---

# Dot Product

![Dot Product](/images/dotproduct.png)

Can only take dot products of vectors that are the same length

# Vector Matrix Multiplication

![Vector Matrix Multiplication](/images/vectormatrixmultiplication.png)

# Matrix Multiplication

![Matrix Multiplication](/images/matrixmultiplication.png)

Final matrix will be of size num rows in AT, num cols in W

Number of columns in first matrix must be the number of rows in second matrix

```py
A = np.array([[1,2,3], [4,5,6]])
W = np.array([[3,5,6,7], [1,2,3,4]])
AT = A.T
Z = np.matmul(AT, W) # alt Z = AT @ W
```
