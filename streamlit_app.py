import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype

# Load dataset
df = pd.read_csv('path_to_your_dataset.csv')
print("Data Sample:")
print(df.head())

### 1. Know Your Data ###

# Basic information
print("\nDataset Information:")
df.info()

print("\nStatistical Summary (Numerical):")
print(df.describe())

# Handling missing values
missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_count, 'Percentage': missing_percentage})
print("\nMissing Values Summary:")
print(missing_df[missing_df['Missing Values'] > 0])

# Visualizing missing values
plt.figure(figsize=(10, 6))
missing_df[missing_df['Percentage'] > 0].plot(kind='bar', y='Percentage', legend=False)
plt.title("Percentage of Missing Values per Column")
plt.ylabel("Percentage")
plt.xlabel("Column")
plt.xticks(rotation=45)
plt.show()

### 2. Feature Engineering ###

# Adding title_length
df['title_length'] = df['title'].apply(len)

# Extracting month from date
df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.month.astype(str)

# Whether the article has a subtitle
df['with_subtitle'] = np.where(df['subtitle'].isnull(), 'No', 'Yes')

# Drop unnecessary columns
df = df.drop(['id', 'subtitle', 'title', 'url', 'date', 'image', 'responses'], axis=1, errors='ignore')

# Populate numeric and categorical lists
num_list = [col for col in df.columns if is_numeric_dtype(df[col])]
cat_list = [col for col in df.columns if is_string_dtype(df[col])]

print("\nNumerical Columns:", num_list)
print("Categorical Columns:", cat_list)

### 3. Univariate Analysis ###

# Plotting categorical variables
for col in cat_list:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index[:10])  # Show top 10
    plt.title(f"Frequency of Top 10 Values in {col}")
    plt.show()

# Plotting numerical variables
for col in num_list:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

### 4. Multivariate Analysis ###

# Correlation heatmap for numerical columns
correlation = df[num_list].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for numerical relationships
sns.pairplot(df[num_list])
plt.suptitle("Pairplot for Numerical Variables", y=1.02)
plt.show()

# Grouped bar chart for categorical vs. categorical relationships
for i, primary_cat in enumerate(cat_list):
    for secondary_cat in cat_list[i+1:]:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=primary_cat, hue=secondary_cat, palette='coolwarm', order=df[primary_cat].value_counts().index[:10])
        plt.title(f"{primary_cat} by {secondary_cat}")
        plt.show()

# Boxplots for numerical vs. categorical relationships
for cat in cat_list:
    for num in num_list:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=cat, y=num, palette="coolwarm")
        plt.title(f"{num} by {cat}")
        plt.xticks(rotation=45)
        plt.show()

# Pairplot with hue for categorical separation in numerical relationships
for hue_cat in cat_list:
    sns.pairplot(df, hue=hue_cat, palette="coolwarm")
    plt.suptitle(f"Pairplot with Hue={hue_cat}", y=1.02)
    plt.show()
