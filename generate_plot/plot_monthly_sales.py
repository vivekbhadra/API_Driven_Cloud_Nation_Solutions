#!/usr/bin/env python3
# plot_monthly_sales.py
# Purpose: Plot total monthly sales trend from EDA CSV output

import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
monthly_sales = pd.read_csv("monthly_sales.csv")

# Plot Monthly Total Sales
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales["month"], monthly_sales["total_sales"], marker="o", color="teal", linewidth=2)
plt.title("Monthly Total Sales Trend", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png", dpi=300)
plt.show()

