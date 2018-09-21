
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_url = 'cluster.csv'

train = pd.read_csv(train_url) # Reading the data using pandas

print(train.columns.values) # Display the train data column's values

print(train.isnull().head()) # Finding the null values in the head

# Missing values with mean column values in the train set

train.fillna(train.mean(), inplace=True) # Fill the missing values in the train dataset


print(train.info()) # Plotting the train data characteristics


X=np.array(train.drop(['X'],axis=1))

#Creating the model

kmeans = KMeans(n_clusters=2,max_iter=100, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
import matplotlib.pyplot as plt

plt.plot(X)

plt.ylabel('Number of groups')

plt.show()