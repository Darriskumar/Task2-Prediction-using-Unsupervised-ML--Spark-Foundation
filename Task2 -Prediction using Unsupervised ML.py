#!/usr/bin/env python
# coding: utf-8

# # Darris Femilia
# # Task2- Spark Foundation
# #  Prediction using Unsupervised ML

# In[1]:


#import libraries

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

#Loading the dataset
iris_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
iris_data


# In[2]:


#display top 5 rows

iris_data.head() 


# In[3]:


#display last 5 rows

iris_data.tail() 


# In[4]:


iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


# In[5]:


#Creating a list of features to be used for clustering

X = iris_data.iloc[:, :-1].values


# In[6]:


#Determining the optimum number of clusters using the elbow method

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[7]:


#Plotting the line graph

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[8]:


#optimum number of clusters

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
     


# In[9]:


#Visualization

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Iris')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


# In[ ]:




