import pandas as pd
import matplotlib.pyplot as plt


# Load real and synthetic data
real_data = pd.read_csv("census.csv")  # Ensure you have this file
synthetic_data = pd.read_csv("synthetic_data.csv")


# Select a numerical column (e.g., "age")
column = "age"

plt.hist(real_data[column], bins=30, alpha=0.5, label="Real Data", color="blue")
plt.hist(synthetic_data[column], bins=30, alpha=0.5, label="Synthetic Data", color="red")
plt.legend()
plt.title(f"Distribution of {column}")
plt.show()
