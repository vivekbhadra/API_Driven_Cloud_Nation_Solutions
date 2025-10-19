#!/usr/bin/env python3
# plot_item_sales.py
# Purpose: Plot top 10 items by total sales from EDA CSV output

import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
item_sales = pd.read_csv("item_sales.csv")

# Sort and select top 10 items
top_items = item_sales.sort_values("total_sales", ascending=False).head(10)

# Plot Top 10 Items by Total Sales
plt.figure(figsize=(8, 5))
plt.bar(top_items["item"].astype(str), top_items["total_sales"], color="seagreen")
plt.title("Top 10 Items by Total Sales", fontsize=14)
plt.xlabel("Item ID")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("top_items_sales.png", dpi=300)
plt.show()

