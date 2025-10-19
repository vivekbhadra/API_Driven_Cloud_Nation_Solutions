#!/usr/bin/env python3
# Purpose: Generate EDA plots from AWS Glue CSV outputs

import pandas as pd
import matplotlib.pyplot as plt

# --- Monthly Sales Trend ---
monthly_sales = pd.read_csv("monthly_sales.csv")
plt.figure(figsize=(10,5))
plt.plot(monthly_sales["month"], monthly_sales["total_sales"], marker="o")
plt.title("Monthly Total Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.savefig("monthly_sales_trend.png")
plt.show()

# --- Store Performance ---
store_avg = pd.read_csv("store_avg.csv")
store_avg = store_avg.sort_values("avg_sales", ascending=False)
plt.figure(figsize=(8,5))
plt.bar(store_avg["store"], store_avg["avg_sales"], color="steelblue")
plt.title("Average Sales by Store")
plt.xlabel("Store ID")
plt.ylabel("Average Sales")
plt.savefig("store_avg_sales.png")
plt.show()

# --- Item Performance ---
item_sales = pd.read_csv("item_sales.csv")
top_items = item_sales.sort_values("total_sales", ascending=False).head(10)
plt.figure(figsize=(8,5))
plt.bar(top_items["item"], top_items["total_sales"], color="seagreen")
plt.title("Top 10 Items by Total Sales")
plt.xlabel("Item ID")
plt.ylabel("Total Sales")
plt.savefig("top_items_sales.png")
plt.show()

