import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Initialize the app
st.title("Interactive EDA with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.dataframe(df.head())

    # Identify numerical and categorical columns
    num_list = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_list = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    st.write("#### Numerical Columns:", num_list)
    st.write("#### Categorical Columns:", cat_list)

    # Sidebar navigation
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Data Cleaning & Descriptive Stats", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    # Data Cleaning & Descriptive Stats
    if analysis_type == "Data Cleaning & Descriptive Stats":
        st.header("Data Cleaning & Descriptive Stats")

        # Missing value handling
        st.subheader("Handle Missing Values")
        missing_option = st.radio(
            "Choose a method to handle missing values:",
            ["Leave as is", "Impute with Mean", "Remove Rows with Missing Data"]
        )
        if missing_option == "Impute with Mean":
            df[num_list] = df[num_list].fillna(df[num_list].mean())
        elif missing_option == "Remove Rows with Missing Data":
            df = df.dropna()

        # Remove duplicates
        st.subheader("Remove Duplicate Rows")
        if st.button("Remove Duplicates"):
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            st.write(f"Removed {before - after} duplicate rows.")

        # Data type correction
        st.subheader("Correct Data Types")
        for col in df.columns:
            dtype_option = st.selectbox(f"Select data type for '{col}':", ["Automatic", "Integer", "Float", "String", "DateTime"])
            if dtype_option == "Integer":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
            elif dtype_option == "Float":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype_option == "String":
                df[col] = df[col].astype(str)
            elif dtype_option == "DateTime":
                df[col] = pd.to_datetime(df[col], errors='coerce')

        st.write("### Cleaned Data:")
        st.dataframe(df.head())

        # Descriptive statistics
        st.header("Descriptive Statistics")
        st.write("#### Summary Statistics:")
        st.write(df.describe(include='all'))
        if st.checkbox("Show Mode"):
            st.write(df.mode().iloc[0])

    # Univariate Analysis
    elif analysis_type == "Univariate Analysis":
        st.header("Univariate Analysis")
        
        # Numerical data
        st.subheader("Numerical Data")
        num_col = st.selectbox("Select a numerical variable:", num_list)
        if num_col:
            chart_type = st.selectbox("Select chart type:", ["Histogram", "Box Plot", "Density Plot", "QQ Plot"])
            fig, ax = plt.subplots()
            if chart_type == "Histogram":
                sns.histplot(df[num_col], kde=True, ax=ax)
            elif chart_type == "Box Plot":
                sns.boxplot(x=df[num_col], ax=ax)
            elif chart_type == "Density Plot":
                sns.kdeplot(df[num_col], fill=True, ax=ax)
            elif chart_type == "QQ Plot":
                stats.probplot(df[num_col], dist="norm", plot=ax)
            st.pyplot(fig)

        # Categorical data
        st.subheader("Categorical Data")
        cat_col = st.selectbox("Select a categorical variable:", cat_list)
        if cat_col:
            chart_type = st.selectbox("Select chart type:", ["Count Plot", "Bar Chart", "Pie Chart"])
            fig, ax = plt.subplots()
            if chart_type == "Count Plot":
                sns.countplot(x=cat_col, data=df, ax=ax)
            elif chart_type == "Bar Chart":
                df[cat_col].value_counts().plot.bar(ax=ax)
            elif chart_type == "Pie Chart":
                df[cat_col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

    # Bivariate Analysis
    elif analysis_type == "Bivariate Analysis":
        st.header("Bivariate Analysis")
        bivar_type = st.selectbox("Select chart type:", ["Scatter Plot", "Correlation Coefficient", "Bar Plot"])
        
        if bivar_type == "Scatter Plot":
            x_var = st.selectbox("Independent Variable (X, numerical):", num_list)
            y_var = st.selectbox("Dependent Variable (Y, numerical):", num_list)
            hue_var = st.selectbox("Grouping Variable (Hue, categorical, optional):", ["None"] + cat_list)
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=x_var,
                y=y_var,
                hue=None if hue_var == "None" else hue_var,
                data=df,
                ax=ax
            )
            ax.set_title(f"Scatter Plot: {y_var} vs {x_var}")
            st.pyplot(fig)

        elif bivar_type == "Correlation Coefficient":
            x_var = st.selectbox("First Variable (X, numerical):", num_list)
            y_var = st.selectbox("Second Variable (Y, numerical):", num_list)
            corr = df[x_var].corr(df[y_var])
            st.write(f"Correlation between {x_var} and {y_var}: {corr:.2f}")

        elif bivar_type == "Bar Plot":
            x_var = st.selectbox("Categorical Variable (X):", cat_list)
            y_var = st.selectbox("Numerical Variable (Y):", num_list)
            fig, ax = plt.subplots()
            sns.barplot(x=x_var, y=y_var, data=df, ax=ax)
            ax.set_title(f"Bar Plot: {y_var} grouped by {x_var}")
            st.pyplot(fig)

    # Multivariate Analysis
    elif analysis_type == "Multivariate Analysis":
        st.header("Multivariate Analysis")
        multivar_type = st.selectbox("Select chart type:", ["Correlation Matrix", "Pair Plot", "Grouped Bar Chart"])
        
        if multivar_type == "Correlation Matrix":
            st.write("### Correlation Matrix (Numerical Variables Only)")
            fig, ax = plt.subplots()
            sns.heatmap(df[num_list].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)

        elif multivar_type == "Pair Plot":
            st.write("### Pair Plot (Numerical Variables Only)")
            sns.pairplot(df[num_list])
            st.pyplot()

        elif multivar_type == "Grouped Bar Chart":
            x_var = st.selectbox("Independent Variable (X, categorical):", cat_list)
            hue_var = st.selectbox("Grouping Variable (Hue, categorical):", cat_list)
            fig, ax = plt.subplots()
            sns.countplot(x=x_var, hue=hue_var, data=df, ax=ax)
            ax.set_title(f"Grouped Bar Chart: {x_var} grouped by {hue_var}")
            st.pyplot(fig)
