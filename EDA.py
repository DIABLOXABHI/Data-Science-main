import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")
data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")


print(customers.info())
print(products.info())
print(transactions.info())
print(data.describe())

print(data.isnull().sum())
top_products = data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(5)
print("Top 5 Products by Quantity:\n", top_products)
region_revenue = data.groupby('Region')['TotalValue'].sum()
print("Revenue by Region:\n", region_revenue)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_products.index, y=top_products.values)
plt.title("Top 5 Products by Quantity")
plt.ylabel("Quantity Sold")
plt.xlabel("Product Name")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10, 5))
sns.barplot(x=region_revenue.index, y=region_revenue.values)
plt.title("Revenue by Region")
plt.ylabel("Total Revenue")
plt.xlabel("Region")
plt.show()
