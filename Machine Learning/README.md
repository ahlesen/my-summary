# About
SkolTech course

The course is a general introduction to machine learning (ML) and its applications. It covers fundamental modern topics in ML and explains key theoretical foundations and tools necessary to study the properties of algorithms and justify their use. It also discusses key aspects of the algorithms' applications, illustrated with real-world problems. We present the most novel theoretical tools and concepts, while trying to be as succinct as possible.

The course beings with an overview of canonical ML applications, problems, learning scenarios and an introduction into theoretical foundations. Then we discuss in depth the fundamental ML algorithms for classification, regression, boosting, etc., their properties and practical applications. The last part of the course is devoted to advanced topics in ML such as metric learning, anomaly detection, active learning, etc.

Within practical sections, we show how to apply these methods to crack various real-world problems. Home assignments include applications of ML to industrial problems, study and derivation of modifications of these algorithms, as well as theoretical exercises.

The students are required to be familiar with key concepts in linear algebra, probability, optimization and calculus.


# Shedule
<p align="center">
  <img src="Shedule ML.png" >
</p>

# Assignments

ML2019**HW01-part1**:
- 1.1-1.3 Numpy/Matplotlib tasks
- 1.4 Visualization of the decision rules of several classifiers applied to artificial 2-dimensional dataset make_moons
- 1.5 Test Random Forests and Support Vector Machines on a trivial [Tic Tac Toe Endgame Dataset](https://datahub.io/machine-learning/tic-tac-toe-endgame)
- 1.6 The goal will be to determine the optimal parameters for two Bagging-Based Forest Ensemble **Regressors** and compare the forests for [Concrete Compressive Strength Dataset](https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set)
- 1.7 Multiclass classification problem for [Glass Classification Data](https://www.kaggle.com/uciml/glass)

ML2019**HW01-part2**:
- Theoretical tasks (Regressions, Bayesian Naive Classifier, Nearest Neighbors, Bootstrap, Decision Tree Leaves, Kernel Regression,Kernel Methods)


ML2019**HW02-part1**:
- Theoretical tasks (Model and feature selection,Boosting: gradient boosting, adaboost, NNs,  Data augmentation)

ML2019**HW02-part2**:
- 2.1 Use random forest to find the imortance of features
- 2.2  On these 20 features train each of the following models
* **Linear Regression**
* **Ridge regression**
* **Random forest**
* **DecisionTree** 
* and test its performance using the **Root Mean Squared Logarithmic Error** (RMSLE).
- 2.3  Implement forward method with early stopping
- Boosting: 2.4 Implement gradient boosting algorithm 2.5 In this task you are asked to compare the **training time** of the **GBDT**, the Gradient Boosted Decision Trees, as implemeted by different popular ML libraries. The dataset you shall use is the [UCI Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). 
- NNs: 2.6 Activation functions      2.7 Backpropagation


ML2019**HW03-part1**:
- 3.1-3.2 Bayesian Models ( Laplace approximation, Hessian matrix,  Monte-Carlo estimate of the integral)
- 3.3-3.4 Gaussian Processes (gaussian elimination, using [**GPy**](https://pypi.python.org/pypi/GPy) library for training and prediction)

ML2019**HW03-part2**:
- 3.1 Practice with Different Clustering Algorithms(`KMeans`, `Gaussian Mixture`, `AgglomerativeClustering`, `Birch`) (use **two** clustering metrics: `silhouette score`
and `adjusted mutual information`,finding the Number of Clusters with Bootstrap)
- 3.2 Dimentionality Reduction and Manifold Learning (theoretical tasks
- 3.3 MNIST principal component analysis (linear  data decomposition with `PCA`; `PCA`, `ICA` and `Isomap` non-linear decompositions;  `KernelPCA`)


ML2019**HW04**(additional):
- Theoretical tasks (SVDD problem,  The Lagrangian and KKT conditions, The dual problem)



# Project 
We aim to do the project on clustering of time series with topological data analysis. There are several works, in which TDA is applied to time series analysis, but still very little research is done on the application of TDA to time series clustering, especially in case of commercial data. We are going to replicate the methodology, done by Lacombe, on two large commercial datasets, in order to cluster time series and assess how the created clusters could have been used in further analysis of data and whether application of TDA in this case is useful.

