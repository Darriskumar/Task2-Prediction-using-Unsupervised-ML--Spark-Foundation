# Task2-Prediction-using-Unsupervised-ML--Spark-Foundation

#importing the necessary libraries: 
pandas for data handling, matplotlib.pyplot for plotting, and the KMeans class from the sklearn.cluster module for K-Means clustering.

#Loading the Dataset:
The Iris dataset is loaded from a URL using pd.read_csv(). It contains information about Iris flowers' sepal and petal measurements.

#Assigning Column Names:
To assign meaningful column names to the dataset using iris_data.columns. These names include 'sepal_length,' 'sepal_width,' 'petal_length,' 'petal_width,' and 'class' for the corresponding features.

#Creating a List of Features:
create a feature list (X) by selecting all columns except the 'class' column. This prepares the data for clustering.

#Optimum Number of Clusters (Elbow Method):
The K-Means clustering with a range of values for k (number of clusters) from 1 to 10. For each value of k, compute within-cluster sum of squares (WCSS) using the KMeans model, which you then append to the wcss list.

#Plotting the Elbow Method:
Use plt.plot() to create a line graph that shows the WCSS for different values of k. The "elbow" of the graph represents the point where the rate of decrease in WCSS starts to slow down. This point helps in determining the optimal number of clusters.

#Optimum Number of Clusters:
Based on the Elbow Method, choosing the optimal number of clusters to be 3 and create a K-Means model with this value. This means it cluster the Iris data into 3 groups.

#Visualization:
The clustered data by creating a scatter plot. To assign different colors to data points belonging to each cluster: red for 'Iris-setosa,' blue for 'Iris-versicolour,' and green for 'Iris-virginica.' Additionally, mark the centroids of the clusters in yellow. Provide appropriate titles, labels, and a legend to make the plot informative and visually appealing.

