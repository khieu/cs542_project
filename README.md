HOW TO RUN CLUSTERING EXPERIMENT:

The code for clustering experiment is inside "Clustering experiments.ipynb".

Specifically:

1. The first cell read the input and pre process data
2. the second cell is implementation for K-fold cross validation 
3. the 3rd cell is the implementation for K-means clustering. K can be set to any number but in the implementation it is set to 3
to fit the phishing data
4. 4th cell: Running K means clustering on the train data. the clustering algorithm is terminated after 4 iteration
and it prints out how many of each cluster is of class "-1", "0" or "1"

From this result, we can calculate how many of each actual class is correctly clustered.

5. 5th cell: Running Gaussian Mixture Model for the training data using sklearn

6. 6th cell: Evaluate how many of each class is misclustered.
 