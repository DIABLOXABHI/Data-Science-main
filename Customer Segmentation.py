import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge datasets
data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")
customer_data = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'Region': 'first', 
}).reset_index()


customer_data_encoded = pd.get_dummies(customer_data, columns=['Region'], drop_first=True)

scaler = StandardScaler()
numerical_features = customer_data_encoded.drop(['CustomerID'], axis=1)
numerical_scaled = scaler.fit_transform(numerical_features)

# Clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(numerical_scaled)
customer_data['Cluster'] = clusters #addding cluster label here

db_index = davies_bouldin_score(numerical_scaled, clusters)
print(f"Davies-Bouldin Index: {db_index}") #checkcing clustering performance


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(numerical_scaled)

plt.figure(figsize=(10, 7)) #evaluation via Principle component analysis
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis')
plt.title("Customer Clusters (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# Saving and summary
customer_data.to_csv("ClusteringResults.csv", index=False)
print("ClusteringResults.csv generated successfully!")
