from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd

real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(32500)
print(synthetic_data)

# Convert Data frame
df = pd.DataFrame(synthetic_data)

# Save to CSV
df.to_csv("synthetic_data.csv", index=False)

print("Synthetic data saved to synthetic_data.csv")