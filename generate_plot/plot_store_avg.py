#!/usr/bin/env python3
# plot_store_avg.py
# Purpose: Plot average sales per store from EDA CSV output

import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
store_avg = pd.read_csv("store_avg.csv")

# Sort by descending average sales
store_avg = store_avg.sort_values("avg_sales", ascending=False)

# Plot Average Sales by Store
plt.figure(figsize=(8, 5))
plt.bar(store_avg["store"], store_avg["avg_sales"], color="steelblue")
plt.title("Average Sales by Store", fontsize=14)
plt.xlabel("Store ID")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("store_avg_sales.png", dpi=300)
plt.show()

