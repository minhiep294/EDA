import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize the app
st.title("EDA with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")

# File upload section
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Identify numerical and categorical columns
    num_list = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_list = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    st.write("Numerical Columns:", num_list)
    st.write("Categorical Columns:", cat_list)

    # Sidebar menu for navigation
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Data Cleaning & Descriptive Stats", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    # Data Cleaning & Descriptive Stats
    if analysis_type == "Data Cleaning & Descriptive Stats":
        st.header("1. Data Cleaning")
        st.subheader("Handle Missing Values")
        missing_option = st.radio(
            "Choose a method to handle missing values:",
            ("Impute with Mean", "Remove Rows with Missing Data", "Leave as is")
        )
        if missing_option == "Impute with Mean":
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_option == "Remove Rows with Missing Data":
            df.dropna(inplace=True)
        
        st.subheader("Remove Duplicates")
        if st.button("Remove Duplicate Rows"):
            before = df.shape[0]
            df.drop_duplicates(inplace=True)
            after = df.shape[0]
            st.write(f"Removed {before - after} duplicate rows")

        st.subheader("Correct Data Types")
        for col in df.columns:
            col_type = st.selectbox(
                f"Select data type for {col}",
                ("Automatic", "Integer", "Float", "String", "DateTime"), index=0
            )
            if col_type == "Integer":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
            elif col_type == "Float":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col_type == "String":
                df[col] = df[col].astype(str)
            elif col_type == "DateTime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
        st.write("Data Cleaning Complete.")
        st.write(df.head())

        st.header("2. Descriptive Statistics")
        st.subheader("Central Tendency & Dispersion")
        st.write(df.describe(include='all'))

        if st.checkbox("Show Mode"):
            st.write(df.mode().iloc[0])

    # Univariate Analysis
    elif analysis_type == "Univariate Analysis":
        st.header("Univariate Analysis")

        st.subheader("Numerical Data Visualization")
        num_col = st.selectbox("Select a numerical variable:", num_list)
        if num_col:
            chart_type = st.selectbox(
                "Select chart type:",
                ["Histogram", "Box Plot", "Density Plot", "Area Plot", "Dot Plot", "Frequency Polygon", "QQ Plot"]
            )

            fig, ax = plt.subplots()
            if chart_type == "Histogram":
                sns.histplot(df[num_col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {num_col}")
            elif chart_type == "Box Plot":
                sns.boxplot(x=df[num_col], ax=ax)
                ax.set_title(f"Box Plot of {num_col}")
            elif chart_type == "Density Plot":
                sns.kdeplot(df[num_col], fill=True, ax=ax)
                ax.set_title(f"Density Plot of {num_col}")
            elif chart_type == "Area Plot":
                sns.histplot(df[num_col], kde=True, fill=True, element="poly", ax=ax)
                ax.set_title(f"Area Plot of {num_col}")
            elif chart_type == "Dot Plot":
                sns.stripplot(x=df[num_col], jitter=True, ax=ax)
                ax.set_title(f"Dot Plot of {num_col}")
            elif chart_type == "Frequency Polygon":
                sns.histplot(df[num_col], kde=False, element="step", color="blue", ax=ax)
                ax.set_title(f"Frequency Polygon of {num_col}")
            elif chart_type == "QQ Plot":
                from scipy import stats
                stats.probplot(df[num_col], dist="norm", plot=ax)
                ax.set_title(f"QQ Plot of {num_col}")
            st.pyplot(fig)

        st.subheader("Categorical Data Visualization")
        cat_col = st.selectbox("Select a categorical variable:", cat_list)
        if cat_col:
            cat_chart_type = st.selectbox(
                "Select chart type:",
                ["Count Plot", "Bar Chart", "Pie Plot"]
            )

            fig, ax = plt.subplots()
            if cat_chart_type == "Count Plot":
                sns.countplot(x=df[cat_col], ax=ax)
                ax.set_title(f"Count Plot of {cat_col}")
            elif cat_chart_type == "Bar Chart":
                sns.barplot(
                    x=df[cat_col].value_counts().index,
                    y=df[cat_col].value_counts().values,
                    ax=ax
                )
                ax.set_title(f"Bar Chart of {cat_col}")
            elif cat_chart_type == "Pie Plot":
                df[cat_col].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_ylabel('')
                ax.set_title(f"Pie Plot of {cat_col}")
            st.pyplot(fig)

    # Bivariate Analysis
    elif analysis_type == "Bivariate Analysis":
        st.header("Bivariate Analysis")
        # Add bivariate analysis code here

    # Multivariate Analysis
    elif analysis_type == "Multivariate Analysis":
        st.header("Multivariate Analysis")
        # Add multivariate analysis code here
