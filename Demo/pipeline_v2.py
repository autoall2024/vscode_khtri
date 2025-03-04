
import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import great_expectations as ge
from ctgan import CTGAN, load_demo
from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Functions for Data pipeline
# Step 1: Load real dataset
def get_real_dataset():
    df = load_demo()
    return df

# Step 2: Train CTGAN and generate synthetic data
def train_ctgan(df, target_col, epochs=10):
    discrete_cols = [col for col in df.columns if col != target_col and df[col].dtype == 'object']
    ctgan = CTGAN(epochs=epochs)
    ctgan.fit(df.drop(columns=[target_col]), discrete_cols)
    return ctgan

def generate_synthetic_data(ctgan_model, num_rows=1000):
    return ctgan_model.sample(num_rows)


def add_noise_to_numeric(df, noise_level=0.05):
    """
    Adds Gaussian noise to numeric columns of a DataFrame.
    
    :param df: A pandas DataFrame
    :param noise_level: Proportion of the column's std dev to use as noise
    :return: A new DataFrame with noise added to numeric columns
    """
    df_noisy = df.copy()
    numeric_cols = df_noisy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        std_dev = df_noisy[col].std()
        # If a column is constant (std=0), skip to avoid NaN issues
        if std_dev > 0:
            noise = np.random.normal(loc=0, scale=noise_level * std_dev, size=df_noisy[col].shape)
            df_noisy[col] = df_noisy[col] + noise
    
    return df_noisy


def shuffle_categorical(df, columns=None):
    """
    Randomly shuffles the specified categorical columns independently.
    If columns is None, all object-type columns will be shuffled.
    """
    df_shuffled = df.copy()
    
    if columns is None:
        # Identify all categorical/object columns
        columns = df_shuffled.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns:
        df_shuffled[col] = np.random.permutation(df_shuffled[col].values)
    
    return df_shuffled


def mix_real_and_synthetic(df_real, df_synthetic):
    """
    Concatenates real and synthetic DataFrames.
    Useful for expanding a small dataset or balancing classes.
    """
    df_mixed = pd.concat([df_real, df_synthetic], ignore_index=True)
    return df_mixed


def augment_synthetic_data(df_synthetic, noise_level=0.05, shuffle_cols=None):
    """
    Applies multiple augmentation techniques to synthetic data:
      1) Add noise to numeric columns
      2) Shuffle specified categorical columns (if any)
    Returns an augmented copy of df_synthetic.
    """
    # 1. Add noise to numeric columns
    df_aug = add_noise_to_numeric(df_synthetic, noise_level=noise_level)

    # 2. Shuffle categorical columns
    df_aug = shuffle_categorical(df_aug, columns=shuffle_cols)
    
    return df_aug


# Step 3: Detect and remove anomalies
def detect_anomalies(df, contamination=0.05, target_col='income'):
    df_copy = df.copy()
    
    # Encode categorical columns
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object' or df_copy[col].dtype.name == 'category':
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))

    # Fit IsolationForest on feature columns only
    clf = IForest(contamination=contamination)
    clf.fit(df_copy.drop(columns=[target_col]).values)
    
    # Return the cleaned dataset with anomalies removed
    mask_normal = (clf.labels_ == 0)
    return df[mask_normal].copy()

# Step 4: Validate data using Great Expectations
def validate_data(df, target_col):
    ge_df = ge.from_pandas(df)
    ge_df.expect_column_values_to_not_be_null(target_col)
    ge_df.expect_column_values_to_be_in_set(target_col, df[target_col].unique())
    result = ge_df.validate()
    return result

# Step 5: Train and evaluate a classifier
# Using LabelEncoder
def train_and_evaluate(df, target_col):
    df_copy = df.copy()
    
    # Label Encode categorical columns
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object' or df_copy[col].dtype.name == 'category':
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    
    # Split features and target
    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return accuracy


# Using OneHotEncoder
# def train_and_evaluate(df, target_col):
#     # One-Hot Encode categorical columns
#     df_encoded = pd.get_dummies(df, drop_first=True)
#     # print(df_encoded.columns)
    
#     # Find the new target column after one-hot encoding
#     new_target_col = [col for col in df_encoded.columns if col.startswith(target_col)][0]  # Example: 'income_>50K'
    
#     # Split features and target
#     X = df_encoded.drop(columns=[new_target_col])
#     y = df_encoded[new_target_col]
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train the model
#     clf = RandomForestClassifier()
#     clf.fit(X_train, y_train)
    
#     # Evaluate accuracy
#     accuracy = accuracy_score(y_test, clf.predict(X_test))
#     return accuracy


# Functions for Data statistics (mean, variance, and standard deviation) comparation
def calculate_statistics(df, label):
    """Calculate mean, variance, and standard deviation for each numeric column."""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    
    stats = pd.DataFrame({
        'Mean': numeric_cols.mean(),
        'Variance': numeric_cols.var(),
        'Standard Deviation': numeric_cols.std()
    })
    stats['Dataset'] = label

    # print("\nStatistics Calculation Results:")    # uncomment for debug
    # print(stats.head())   # uncomment for debug
    
    return stats


# def plot_statistics(real_stats, synthetic_stats, cleaned_stats):
#     """Plot mean, variance, and standard deviation comparisons."""
#     stats = pd.concat([real_stats, synthetic_stats, cleaned_stats])
#     stats = stats.reset_index().melt(id_vars=['index', 'Dataset'], var_name='Metric', value_name='Value')
    
#     plt.figure(figsize=(15, 6))
#     sns.barplot(data=stats, x='index', y='Value', hue='Dataset', ci=None)
#     plt.xticks(rotation=45)
#     plt.title('Comparison of Mean, Variance, and Standard Deviation')
#     plt.xlabel('Features')
#     plt.ylabel('Value')
#     plt.legend(title='Dataset')
#     plt.tight_layout()
#     plt.show()


def plot_correlation_heatmap(df, title):
    """Plot a heatmap of the correlation matrix for a dataset."""
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Correlation Heatmap - {title}')
    # plt.show()
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{title}_corr_heatmap.png')
    plt.savefig(file_path, format='png')
    print(f"Plot saved as: {file_path}")
    plt.close()  # Close the plot to avoid overlapping plots when generating multiple figures


def plot_metric_comparison(real_stats, synthetic_stats, cleaned_stats, metric):
    """Plot a separate bar chart for a specific metric (Mean, Variance, or Standard Deviation)."""
    # Combine statistics for the specified metric
    stats = pd.concat([
        real_stats[['Mean', 'Variance', 'Standard Deviation', 'Dataset']].assign(Metric=metric),
        synthetic_stats[['Mean', 'Variance', 'Standard Deviation', 'Dataset']].assign(Metric=metric),
        cleaned_stats[['Mean', 'Variance', 'Standard Deviation', 'Dataset']].assign(Metric=metric)
    ]).reset_index().rename(columns={"index": "Feature"})
    
    plt.figure(figsize=(15, 6))
    sns.barplot(data=stats, x='Feature', y=metric, hue='Dataset', ci=None)
    plt.xticks(rotation=45)
    plt.title(f'Comparison of {metric}')
    plt.xlabel('Features')
    plt.ylabel(metric)
    plt.legend(title='Dataset')
    plt.tight_layout()
    # plt.show()
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{metric}_comparation.png')
    plt.savefig(file_path, format='png')
    print(f"Plot saved as: {file_path}")
    plt.close()  # Close the plot to avoid overlapping plots when generating multiple figures


def plot_statistics(real_stats, synthetic_stats, cleaned_stats):
    """Plot mean, variance, and standard deviation as separate figures."""
    print("Plotting Mean Comparison...")
    plot_metric_comparison(real_stats, synthetic_stats, cleaned_stats, 'Mean')

    print("Plotting Variance Comparison...")
    plot_metric_comparison(real_stats, synthetic_stats, cleaned_stats, 'Variance')

    print("Plotting Standard Deviation Comparison...")
    plot_metric_comparison(real_stats, synthetic_stats, cleaned_stats, 'Standard Deviation')


# Functions for Visual data distributions 
# for both numerical and categorical columns for the real, synthetic, and cleaned datasets
def plot_numerical_distribution(df_real, df_synthetic, df_cleaned, column):
    """Plot the distribution of a numerical column for real, synthetic, and cleaned datasets."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df_real[column], label='Real Data', kde=True, color='blue', stat='density', alpha=0.5)
    sns.histplot(df_synthetic[column], label='Synth-Aug Data', kde=True, color='green', stat='density', alpha=0.5)
    sns.histplot(df_cleaned[column], label='Cleaned Data', kde=True, color='red', stat='density', alpha=0.5)
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    # plt.show()
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{column}_distribution.png')
    plt.savefig(file_path, format='png')
    print(f"Plot saved as: {file_path}")
    plt.close()  # Close the plot to avoid overlapping plots when generating multiple figures


def plot_categorical_distribution(df_real, df_synthetic, df_cleaned, column):
    """Plot the distribution of a categorical column for real, synthetic, and cleaned datasets."""
    real_counts = df_real[column].value_counts(normalize=True)
    synthetic_counts = df_synthetic[column].value_counts(normalize=True)
    cleaned_counts = df_cleaned[column].value_counts(normalize=True)
    
    combined_counts = pd.DataFrame({
        'Real Data': real_counts,
        'Synth-Aug Data': synthetic_counts,
        'Cleaned Data': cleaned_counts
    }).fillna(0)
    
    combined_counts.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{column}_distribution.png')
    plt.savefig(file_path, format='png')
    print(f"Plot saved as: {file_path}")
    plt.close()  # Close the plot to avoid overlapping plots when generating multiple figures


# Main functions for Data visualization
def visualize_comparison(df_real, df_synthetic, df_cleaned):
    # Calculate statistics for each dataset
    print("Calculating statistics...")
    real_stats = calculate_statistics(df_real, label='Real Data')
    synthetic_stats = calculate_statistics(df_synthetic, label='Synth-Aug Data')
    cleaned_stats = calculate_statistics(df_cleaned, label='Cleaned Data')
    print("Calculating statistics done.")

    # Plot statistics comparison
    print("\nPlotting statistics comparison...")
    plot_statistics(real_stats, synthetic_stats, cleaned_stats)
    print("Plotting statistics comparison done.")

    # Plot correlation heatmaps for each dataset
    print("\nPlotting correlation heatmaps...")
    plot_correlation_heatmap(df_real, 'Real Data')
    plot_correlation_heatmap(df_synthetic, 'Synth-Aug Data')
    plot_correlation_heatmap(df_cleaned, 'Cleaned Data')
    print("Plotting correlation heatmaps done.")


def visualize_distributions(df_real, df_synthetic, df_cleaned):
    """Visualize distributions for both numerical and categorical columns."""
    numeric_columns = df_real.select_dtypes(include=[np.number]).columns
    categorical_columns = df_real.select_dtypes(include=['object', 'category']).columns
    
    print("\nVisualizing numerical distributions...")
    for column in numeric_columns:
        plot_numerical_distribution(df_real, df_synthetic, df_cleaned, column)
    print("Visualizing numerical distributions done.")
    
    print("\nVisualizing categorical distributions...")
    for column in categorical_columns:
        plot_categorical_distribution(df_real, df_synthetic, df_cleaned, column)
    print("Visualizing categorical distributions done.")


# Main Pipeline Execution
def run_pipeline():
    print("\n==== Starting Pipeline ====")

    # Step 1: Load dataset
    print("\nStep 1: Loading real dataset...")
    df_real = get_real_dataset()
    print(f"Real dataset shape: {df_real.shape}")

    # Step 2: Train CTGAN and generate synthetic data
    print("\nStep 2: Training CTGAN and generating synthetic data...")
    ctgan_model = train_ctgan(df_real, target_col='income', epochs=300)  # more epochs generally lead to better synthetic data but take more time
    df_synthetic = generate_synthetic_data(ctgan_model, num_rows=5000)   # more num_rows will generate more synthetic data
    print(f"Synth-Aug dataset shape: {df_synthetic.shape}")

    # ------------------------------------------------------------------------------
    # ADDED CODE: Apply AI-based data augmentation to the synthetic data
    # ------------------------------------------------------------------------------
    
    # Example 2A: Add noise + shuffle categorical columns
    # (Identify which columns you want to shuffle; here we pick a few known categorical columns 
    #  from the Adult dataset. Adjust as needed.)
    shuffle_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']
    
    df_synthetic_aug = augment_synthetic_data(
        df_synthetic, 
        noise_level=0.05,        # Adjust the noise level as needed
        shuffle_cols=shuffle_cols
    )
    
    # Example 2B (Optional): Combine the original synthetic data with the augmented one 
    # to expand your synthetic dataset further
    df_synthetic_aug = pd.concat([df_synthetic, df_synthetic_aug], ignore_index=True)
    
    # Example 2C (Optional): Mix real data and synthetic data if you want one combined dataset
    df_synthetic_aug = mix_real_and_synthetic(df_real, df_synthetic_aug)

    # Now df_synthetic_aug holds your "augmented" synthetic data
    # ------------------------------------------------------------------------------

    # Step 3: Detect and remove anomalies
    print("\nStep 3: Detecting anomalies...")
    df_synthetic_aug['income'] = np.random.choice(df_real['income'].unique(), size=len(df_synthetic_aug))  # Ensure target column is present
    df_cleaned = detect_anomalies(df_synthetic_aug, contamination=0.05, target_col='income')
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    # print(df_cleaned['income'].value_counts())    # debugging purpose

    # Step 4: Validate data
    print("\nStep 4: Validating data...")
    validation_result = validate_data(df_cleaned, target_col='income')
    if validation_result.success:
        print("Data validation passed.")
    else:
        print("Data validation failed.")


    # Step 5: Train and evaluate model
    print("\nStep 5: Training and evaluating model...")
    accuracy = train_and_evaluate(df_cleaned, target_col='income')
    print(f"Model accuracy: {accuracy:.2f}")

    print("\n==== Pipeline completed. ====")

    return df_real, df_synthetic_aug, df_cleaned


# Main Data Visualization Execution
def run_data_visualization(df_real, df_synthetic, df_cleaned):
    print("\n==== Post-pipeline Data Visualization ====")

    # Export data to .csv files
    print("\n1. Exporting data to CSV files...")
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)
    df_real.to_csv(f"{os.path.join(output_dir, 'real_data.csv')}", index=False)
    df_synthetic.to_csv(f"{os.path.join(output_dir, 'synthetic_aug_data.csv')}", index=False)
    df_cleaned.to_csv(f"{os.path.join(output_dir, 'clean_data.csv')}", index=False)
    print("Real and synthetic data saved successfully.")

    # Visualizing Data to check and compare whether the metrics 
    # such as mean, variance, deviation and correlation
    # of the real data, synthetic/augmented data and cleaned data are closely match or not
    print("\n2. Visualizing Data...")
    visualize_comparison(df_real, df_synthetic, df_cleaned)
    visualize_distributions(df_real, df_synthetic, df_cleaned)

    print("\n==== Data Visualization Completed. ====")


def main():
    # Ignoring unwanted warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    # warnings.filterwarnings("ignore", category=FutureWarning)

    # Executing main data pipeline
    df_real, df_synthetic_aug, df_cleaned = run_pipeline()

    # Executing main data visualization
    run_data_visualization(df_real, df_synthetic_aug, df_cleaned)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\nTotal exec time: {exec_time // 3600} hr(s), {exec_time % 3600 // 60} min(s), {exec_time % 3600 % 60} sec(s)")
