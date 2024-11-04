import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the Streamlit page configuration
st.set_page_config(page_title="EDA Visualization App", layout="wide")

# Title and Instructions
st.title("Exploratory Data Analysis Visualization App")
st.write("""
This app helps you create essential charts for single-variable and two-variable analysis.
Each chart has specific requirements for better usability and flexibility across different datasets.
""")

# Upload Data
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Dataset Overview")
    st.dataframe(df)

    # Section 1: Single-Variable Charts
    st.write("## Section 1: Single-Variable Charts")
    st.sidebar.subheader("Single-Variable Charts")

    # 1. Histogram
    st.write("### Histogram")
    st.write("**Requirements:** A continuous variable. Useful for observing the distribution of values.")
    hist_col = st.selectbox("Select Column for Histogram", df.select_dtypes(include=["float", "int"]).columns)
    if hist_col:
        fig, ax = plt.subplots()
        sns.histplot(df[hist_col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {hist_col}")
        st.pyplot(fig)

    # 2. Bar Chart
    st.write("### Bar Chart")
    st.write("**Requirements:** A categorical variable. Useful for observing frequency counts of categories.")
    bar_col = st.selectbox("Select Column for Bar Chart", df.select_dtypes(include=["object", "category"]).columns)
    if bar_col:
        fig, ax = plt.subplots()
        df[bar_col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Bar Chart of {bar_col}")
        st.pyplot(fig)

    # 3. Box Plot
    st.write("### Box Plot")
    st.write("**Requirements:** A continuous variable. Useful for detecting outliers and understanding distribution.")
    box_col = st.selectbox("Select Column for Box Plot", df.select_dtypes(include=["float", "int"]).columns)
    if box_col:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[box_col], ax=ax)
        ax.set_title(f"Box Plot of {box_col}")
        st.pyplot(fig)

    # Section 2: Two-Variable Charts
    st.write("## Section 2: Two-Variable Charts")
    st.sidebar.subheader("Two-Variable Charts")

    # 1. Scatter Plot
    st.write("### Scatter Plot")
    st.write("**Requirements:** Two continuous variables. Useful for visualizing relationships or correlations.")
    scatter_x = st.selectbox("Select X-axis for Scatter Plot", df.select_dtypes(include=["float", "int"]).columns, key="scatter_x")
    scatter_y = st.selectbox("Select Y-axis for Scatter Plot", df.select_dtypes(include=["float", "int"]).columns, key="scatter_y")
    if scatter_x and scatter_y:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[scatter_x], y=df[scatter_y], ax=ax)
        ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
        st.pyplot(fig)

    # 2. Heatmap
    st.write("### Correlation Heatmap")
    st.write("**Requirements:** At least two continuous variables. Useful for examining correlations between multiple variables.")
    if len(df.select_dtypes(include=["float", "int"]).columns) >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # 3. Line Plot
    st.write("### Line Plot")
    st.write("**Requirements:** Suitable for time series data or variables where trends over an index or category are to be observed.")
    line_x = st.selectbox("Select X-axis for Line Plot", df.columns, key="line_x")
    line_y = st.selectbox("Select Y-axis for Line Plot", df.select_dtypes(include=["float", "int"]).columns, key="line_y")
    if line_x and line_y:
        fig, ax = plt.subplots()
        sns.lineplot(x=df[line_x], y=df[line_y], ax=ax)
        ax.set_title(f"Line Plot of {line_y} over {line_x}")
        st.pyplot(fig)

else:
    st.write("Please upload a dataset to start the EDA.")
