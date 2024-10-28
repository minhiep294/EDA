import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Title and Purpose of Analysis
st.title("📊 EDA Dashboard")

# Helper function to format large numbers in an effective way
def format_large_numbers(value):
    if value >= 1_000_000_000:
        return f'{value / 1_000_000_000:.1f}B'  # Billions
    elif value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'      # Millions
    else:
        return f'{value}'                       # Original value if small

# Load dataset uploaded by the user
uploaded_data_file = st.file_uploader("Upload a dataset (.csv)", type=["csv"])

if uploaded_data_file:
    # Read the dataset
    data = pd.read_csv(uploaded_data_file)

    # Display raw data preview
    st.subheader("Raw Data Preview")
    st.write(data.head())  # Show first few rows of the uploaded dataset

    # Ensure the 'Year' column is numeric without commas
    if 'Year' in data.columns:
        # Convert 'Year' to integer and handle any non-numeric values gracefully
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce').astype('Int64')

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write("The table below provides summary statistics for all numeric columns.")
    st.write(data.describe())

    # Table 1: Count of Vietnamese banks over time by classification
    # Ensure 'Year' and 'Classification' columns exist
    if 'Year' in data.columns and 'Classification' in data.columns:
        bank_counts_over_time = data.groupby(['Year', 'Classification']).size().unstack(fill_value=0)
        st.subheader("Number of Vietnamese Banks Over Time")
        st.write(bank_counts_over_time)

    # Table 2: List of Vietnamese Banks covered in the database with full name, bank code, and classification
    if 'Bank Full Name' in data.columns and 'Bank Code' in data.columns and 'Classification' in data.columns:
        bank_list = data[['Bank Full Name', 'Bank Code', 'Classification']].drop_duplicates().reset_index(drop=True)
        st.subheader("List of Vietnamese Banks in Database")
        st.write(bank_list)

    # Sidebar filters for Year and Classification
    st.sidebar.title("Filter Options")
    
    if 'Year' in data.columns and 'Classification' in data.columns:
        years = sorted(data['Year'].dropna().unique())
        classifications = data['Classification'].dropna().unique()

        selected_year = st.sidebar.multiselect("Select Year(s)", years, default=years)
        selected_classification = st.sidebar.multiselect("Select Classification(s)", classifications, default=classifications)

        # Apply filters
        data_filtered = data[data['Year'].isin(selected_year) & data['Classification'].isin(selected_classification)]
    else:
        data_filtered = data

    # Visualization section
    st.subheader("Visualize Your Data")
    
    # Top 5 banks with highest number of branches
    if 'Bank Code' in data_filtered.columns and 'Number of Branches' in data_filtered.columns:
        top_branches = data_filtered.groupby('Bank Code')['Number of Branches'].sum().nlargest(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_branches.index, y=top_branches.values, ax=ax)
        ax.set_title("Top 5 Banks with Highest Number of Branches")
        ax.set_xlabel("Bank Code")
        ax.set_ylabel("Number of Branches")
        st.pyplot(fig)

    # Top 10 banks with highest total assets
    if 'Bank Code' in data_filtered.columns and 'Total Assets' in data_filtered.columns:
        top_assets = data_filtered.groupby('Bank Code')['Total Assets'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_assets.index, y=top_assets.values, ax=ax)
        ax.set_title("Top 10 Banks with Highest Total Assets (Million VND)")
        ax.set_xlabel("Bank Code")
        ax.set_ylabel("Total Assets (Million VND)")
        st.pyplot(fig)

    # Pie chart with Percentage of total assets by classification
    if 'Classification' in data_filtered.columns and 'Total Assets' in data_filtered.columns:
        assets_by_classification = data_filtered.groupby('Classification')['Total Assets'].sum()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(assets_by_classification, labels=assets_by_classification.index, autopct='%1.1f%%', startangle=140)
        ax.set_title("Percentage of Total Assets by Classification (Million VND)")
        st.pyplot(fig)

    # Top 10 banks with highest profits after tax
    if 'Bank Code' in data_filtered.columns and 'Profits After Tax' in data_filtered.columns:
        top_profits = data_filtered.groupby('Bank Code')['Profits After Tax'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_profits.index, y=top_profits.values, ax=ax)
        ax.set_title("Top 10 Banks with Highest Profits After Tax (Million VND)")
        ax.set_xlabel("Bank Code")
        ax.set_ylabel("Profits After Tax (Million VND)")
        st.pyplot(fig)

    # Top 10 banks with highest NPL (Non-Performing Loans)
    if 'Bank Code' in data_filtered.columns and 'Non-performing Loans' in data_filtered.columns:
        top_npl = data_filtered.groupby('Bank Code')['Non-performing Loans'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_npl.index, y=top_npl.values, ax=ax)
        ax.set_title("Top 10 Banks with Highest Non-Performing Loans (NPL) (Million VND)")
        ax.set_xlabel("Bank Code")
        ax.set_ylabel("Non-Performing Loans (NPL) (Million VND)")
        st.pyplot(fig)

    # Top 10 banks with highest ROA (Returns Over Assets)
    if 'Bank Code' in data_filtered.columns and 'Returns Over Assets' in data_filtered.columns:
        top_roa = data_filtered.groupby('Bank Code')['Returns Over Assets'].mean().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=top_roa.index, y=top_roa.values, marker="o", ax=ax)
        ax.set_title("Top 10 Banks with Highest ROA")
        ax.set_xlabel("Bank Code")
        ax.set_ylabel("Returns Over Assets (ROA)")
        st.pyplot(fig)

    # Top 10 banks with highest ROE (Returns Over Equity)
    if 'Bank Code' in data_filtered.columns and 'Returns Over Equity' in data_filtered.columns:
        top_roe = data_filtered.groupby('Bank Code')['Returns Over Equity'].mean().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=top_roe.index, y=top_roe.values, marker="o", ax=ax)
        ax.set_title("Top 10 Banks with Highest ROE")
        ax.set_xlabel("Bank Code")
        ax.set_ylabel("Returns Over Equity (ROE)")
        st.pyplot(fig)

    # Generalized Charting Section
    st.subheader("Custom Chart Generator")

    # Select the type of chart
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])

    # Select x and y variables
    available_columns = list(data_filtered.columns)
    x_var = st.selectbox("Select X-axis variable", available_columns)
    y_var = st.selectbox("Select Y-axis variable (Top 10 based)", available_columns)

    # Convert y_var column to numeric if necessary
    if y_var in data_filtered.columns:
        data_filtered[y_var] = pd.to_numeric(data_filtered[y_var], errors='coerce')  # Converts non-numeric to NaN
    
    # Filter to Top 10 Banks based on selected y variable
    top_10_data = data_filtered.groupby('Bank Code')[y_var].sum().nlargest(10).reset_index()

    # Generate charts based on selection
    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "Bar Chart":
        sns.barplot(x=x_var, y=y_var, data=top_10_data, ax=ax)
        ax.set_title(f"Top 10 Banks by {y_var} (Bar Chart)")
        ax.set_xlabel(x_var)
        ax.set_ylabel(f"{y_var} (Million VND)" if y_var in ["Total Assets", "Profits After Tax", "Non-performing Loans"] else y_var)
    
    elif chart_type == "Line Chart":
        sns.lineplot(x=x_var, y=y_var, data=top_10_data, marker="o", ax=ax)
        ax.set_title(f"Top 10 Banks by {y_var} (Line Chart)")
        ax.set_xlabel(x_var)
        ax.set_ylabel(f"{y_var} (Million VND)" if y_var in ["Total Assets", "Profits After Tax", "Non-performing Loans"] else y_var)

    elif chart_type == "Pie Chart":
        # Pie charts only work with aggregate data (e.g., total assets by classification)
        if y_var == "Total Assets" and "Classification" in available_columns:
            pie_data = data_filtered.groupby("Classification")[y_var].sum()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
            ax.set_title(f"Percentage of {y_var} by Classification (Million VND)")
        else:
            st.warning("Pie charts require a categorical aggregation, such as 'Classification' with 'Total Assets'.")

    # Display the chart
    st.pyplot(fig)

    # Sidebar for variable selection
    st.sidebar.title("Variable Selection")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_columns = st.sidebar.multiselect("Select Variables for Pair Plot", numeric_columns, default=numeric_columns[:3])

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    heatmap_fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data[selected_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(heatmap_fig)
