---
title: Unsupervised Learning, Recommenders, Reinforcement Learning
date: 2022-01-15
description: Coursera ML Specialization Course 3 Notes
category: notes
type: notes
---

### Clustering

Find datapoints that are related to each other

- group similar news
- market segmentation

##### K-Means Clustering

- assign K random cluster centroids
- go through each example and check which cluster centroid it is closer to (mink((xi-uk)^2))
- recompute the centroid by taking the average position of each cluster. If a cluster has no points, eliminate the cluster.
- repeat until the centroids/examples no longer changes

##### Optimization Objective

![K-means optimization](/images/kmeansoptimization.png)

aka as the distortion function

### Anomaly Detection

### Recommender Systems

### Reinforcement Learning

efos34sd92
