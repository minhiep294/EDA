import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype

st.title("Exploratory Data Analysis (EDA) Tool")
st.write("Upload a CSV file to start the EDA process.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Data Sample:")
    st.dataframe(df.head())
    
    ### 1. Know Your Data ###
    
    # Basic information
    st.write("\nDataset Information:")
    buffer = df.info(buf=None)  # Streamlit doesn’t natively display `df.info()`
    st.text(buffer)

    st.write("\nStatistical Summary (Numerical):")
    st.write(df.describe())
    
    # Handling missing values
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_count, 'Percentage': missing_percentage})
    st.write("\nMissing Values Summary:")
    st.write(missing_df[missing_df['Missing Values'] > 0])

    # Visualizing missing values
    if missing_percentage.any():
        plt.figure(figsize=(10, 6))
        missing_df[missing_df['Percentage'] > 0].plot(kind='bar', y='Percentage', legend=False)
        plt.title("Percentage of Missing Values per Column")
        plt.ylabel("Percentage")
        plt.xlabel("Column")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    ### 2. Feature Engineering ###
    
    # Adding title_length
    if 'title' in df.columns:
        df['title_length'] = df['title'].apply(len)
    
    # Extracting month from date
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.month.astype(str)
    
    # Whether the article has a subtitle
    if 'subtitle' in df.columns:
        df['with_subtitle'] = np.where(df['subtitle'].isnull(), 'No', 'Yes')
    
    # Drop unnecessary columns
    drop_cols = ['id', 'subtitle', 'title', 'url', 'date', 'image', 'responses']
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    
    # Populate numeric and categorical lists
    num_list = [col for col in df.columns if is_numeric_dtype(df[col])]
    cat_list = [col for col in df.columns if is_string_dtype(df[col])]
    
    st.write("\nNumerical Columns:", num_list)
    st.write("Categorical Columns:", cat_list)
    
    ### 3. Univariate Analysis ###
    
    # Plotting categorical variables
    for col in cat_list:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, y=col, order=df[col].value_counts().index[:10])  # Show top 10
        plt.title(f"Frequency of Top 10 Values in {col}")
        st.pyplot(plt)

    # Plotting numerical variables
    for col in num_list:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        st.pyplot(plt)

    ### 4. Multivariate Analysis ###
    
    # Correlation heatmap for numerical columns
    if len(num_list) > 1:
        correlation = df[num_list].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        st.pyplot(plt)
    
    # Pairplot for numerical relationships
    if len(num_list) > 1:
        sns.pairplot(df[num_list])
        plt.suptitle("Pairplot for Numerical Variables", y=1.02)
        st.pyplot(plt)
    
    # Grouped bar chart for categorical vs. categorical relationships
    for i, primary_cat in enumerate(cat_list):
        for secondary_cat in cat_list[i+1:]:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=primary_cat, hue=secondary_cat, palette='coolwarm', order=df[primary_cat].value_counts().index[:10])
            plt.title(f"{primary_cat} by {secondary_cat}")
            st.pyplot(plt)
    
    # Boxplots for numerical vs. categorical relationships
    for cat in cat_list:
        for num in num_list:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=cat, y=num, palette="coolwarm")
            plt.title(f"{num} by {cat}")
            plt.xticks(rotation=45)
            st.pyplot(plt)
    
    # Pairplot with hue for categorical separation in numerical relationships
    for hue_cat in cat_list:
        sns.pairplot(df, hue=hue_cat, palette="coolwarm")
        plt.suptitle(f"Pairplot with Hue={hue_cat}", y=1.02)
        st.pyplot(plt)
else:
    st.write("Please upload a dataset to start the analysis.")
