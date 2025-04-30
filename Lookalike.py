import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")
customer_data = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': lambda x: list(x)
}).reset_index()

# Encoding in into one-hot vectors
product_matrix = pd.get_dummies(data.set_index('CustomerID')['ProductID']).groupby('CustomerID').sum()

scaler = StandardScaler()
numerical_features = customer_data[['TotalValue', 'Quantity']]
numerical_scaled = scaler.fit_transform(numerical_features)
final_features = pd.concat([pd.DataFrame(numerical_scaled, index=customer_data['CustomerID']), product_matrix], axis=1)
similarity = cosine_similarity(final_features)
similarity_df = pd.DataFrame(similarity, index=customer_data['CustomerID'], columns=customer_data['CustomerID'])

# Generate Lookalike.csv
lookalike = {}
for customer in similarity_df.index[:20]:  # first 20 cust
    top_matches = similarity_df.loc[customer].sort_values(ascending=False)[1:4]  # top 3 matche
    lookalike[customer] = [(idx, score) for idx, score in zip(top_matches.index, top_matches.values)]

# Saving results to CSV
lookalike_df = pd.DataFrame([(cust, match[0], match[1]) for cust, matches in lookalike.items() for match in matches],columns=['CustomerID', 'LookalikeCustomerID', 'Score'])
lookalike_df.to_csv('Lookalike.csv', index=False)
print("Lookalike.csv generated successfully!")
