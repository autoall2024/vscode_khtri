import pandas as pd

# Load real and synthetic data
real_data = pd.read_csv("census.csv")  # Ensure you have this file
synthetic_data = pd.read_csv("synthetic_data.csv")

# Display basic info
print("Real Data:")
print(real_data.head())

print("\nSynthetic Data:")
print(synthetic_data.head())


# Convert Data frame
df1 = pd.DataFrame(real_data.head())
df2 = pd.DataFrame(synthetic_data.head())

# Save to CSV
df1.to_csv("real_data_head.csv", index=False)
df2.to_csv("synthetic_data_head.csv", index=False)


