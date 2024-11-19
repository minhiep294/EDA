import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Section: Data Cleaning and Descriptive Statistics
def data_cleaning_and_descriptive(df):
    # Section 1: Data Cleaning
    st.header("1. Data Cleaning")

    # Handle Missing Values
    st.subheader("Handle Missing Values")
    missing_option = st.radio("Choose a method to handle missing values:", 
                               ("Leave as is", "Impute with Mean (Numerical Only)", "Remove Rows with Missing Data"))
    if missing_option == "Impute with Mean (Numerical Only)":
        # Impute missing values with mean for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        st.write("Missing values in numerical columns were imputed with the mean.")
    elif missing_option == "Remove Rows with Missing Data":
        before = df.shape[0]
        df.dropna(inplace=True)
        after = df.shape[0]
        st.write(f"Removed {before - after} rows containing missing data.")

    # Remove Duplicates
    st.subheader("Remove Duplicates")
    if st.button("Remove Duplicate Rows"):
        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        st.write(f"Removed {before - after} duplicate rows.")

    # Correct Data Types
    st.subheader("Correct Data Types")
    for col in df.columns:
        col_type = st.selectbox(
            f"Select data type for column: {col}",
            ("Automatic", "Integer", "Float", "String", "DateTime"),
            index=0,
        )
        try:
            if col_type == "Integer":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif col_type == "Float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif col_type == "String":
                df[col] = df[col].astype(str)
            elif col_type == "DateTime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception as e:
            st.write(f"Could not convert {col} to {col_type}: {e}")
    st.write("Data Cleaning Complete.")
    st.write(df.head())

    # Section 2: Descriptive Statistics
    st.header("2. Descriptive Statistics")
    st.write(df.describe(include="all"))

# Dynamic Filter Function
def filter_data(df):
    st.sidebar.title("Filter Data")
    filters = {}
    filter_container = st.sidebar.container()
    with filter_container:
        # Add new filter dynamically
        add_filter_button = st.button("Add New Filter")
        if "filter_count" not in st.session_state:
            st.session_state.filter_count = 0

        if add_filter_button:
            st.session_state.filter_count += 1

        # Manage filters dynamically
        for i in range(st.session_state.filter_count):
            with st.expander(f"Filter {i+1}", expanded=True):
                col_name = st.selectbox(
                    f"Select Column for Filter {i+1}",
                    df.columns,
                    key=f"filter_col_{i}",
                )
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    min_val = int(df[col_name].min())  # Convert to integer
                    max_val = int(df[col_name].max())  # Convert to integer
                    selected_range = st.slider(
                        f"Select range for {col_name}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"filter_slider_{i}",
                    )
                    filters[col_name] = ("range", selected_range)
                elif pd.api.types.is_string_dtype(df[col_name]):
                    unique_values = df[col_name].unique().tolist()
                    selected_values = st.multiselect(
                        f"Select categories for {col_name}",
                        options=unique_values,
                        default=unique_values,
                        key=f"filter_multiselect_{i}",
                    )
                    filters[col_name] = ("categories", selected_values)
                elif pd.api.types.is_datetime64_any_dtype(df[col_name]):
                    min_date = df[col_name].min()
                    max_date = df[col_name].max()
                    selected_dates = st.date_input(
                        f"Select date range for {col_name}",
                        value=(min_date, max_date),
                        key=f"filter_date_{i}",
                    )
                    filters[col_name] = ("dates", selected_dates)

                # Remove filter
                remove_filter = st.button(f"Remove Filter {i+1}", key=f"remove_filter_{i}")
                if remove_filter:
                    del st.session_state[f"filter_col_{i}"]
                    del st.session_state[f"filter_slider_{i}"]
                    del st.session_state[f"filter_multiselect_{i}"]
                    del st.session_state[f"filter_date_{i}"]
                    st.session_state.filter_count -= 1
                    break

    # Apply filters to the dataset
    filtered_df = df.copy()
    for col, (filter_type, value) in filters.items():
        if filter_type == "range":
            filtered_df = filtered_df[
                (filtered_df[col] >= value[0]) & (filtered_df[col] <= value[1])
            ]
        elif filter_type == "categories":
            filtered_df = filtered_df[filtered_df[col].isin(value)]
        elif filter_type == "dates":
            filtered_df = filtered_df[
                (filtered_df[col] >= pd.to_datetime(value[0]))
                & (filtered_df[col] <= pd.to_datetime(value[1]))
            ]

    return filtered_df

# Univariate Analysis
def univariate_analysis(df, num_list, cat_list):
    st.subheader("Univariate Analysis")
    variable_type = st.radio("Choose variable type:", ["Numerical", "Categorical"])
    
    # Numerical Variable Analysis
    if variable_type == "Numerical":
        col = st.selectbox("Select a numerical variable:", num_list)
        chart_type = st.selectbox("Choose chart type:", ["Histogram", "Box Plot", "Density Plot", "QQ Plot"])
        fig, ax = plt.subplots()
        if chart_type == "Histogram":
            sns.histplot(df[col], kde=True, ax=ax)
        elif chart_type == "Box Plot":
            sns.boxplot(x=df[col], ax=ax)
        elif chart_type == "Density Plot":
            sns.kdeplot(df[col], fill=True, ax=ax)
        elif chart_type == "QQ Plot":
            stats.probplot(df[col], dist="norm", plot=ax)
        ax.set_title(f"{chart_type} for {col}")
        st.pyplot(fig)
    
    # Categorical Variable Analysis
    elif variable_type == "Categorical":
        col = st.selectbox("Select a categorical variable:", cat_list)
        chart_type = st.selectbox("Choose chart type:", ["Count Plot", "Bar Chart", "Pie Chart", "Box Plot"])
        fig, ax = plt.subplots()
        if chart_type == "Count Plot":
            sns.countplot(x=col, data=df, ax=ax)
        elif chart_type == "Bar Chart":
            df[col].value_counts().plot.bar(ax=ax)
        elif chart_type == "Pie Chart":
            df[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
        elif chart_type == "Box Plot":
            num_col = st.selectbox("Select a numerical variable for Box Plot:", num_list)
            sns.boxplot(x=col, y=num_col, data=df, ax=ax)
        ax.set_title(f"{chart_type} for {col}")
        st.pyplot(fig)

# Bivariate Analysis
def bivariate_analysis(df, num_list, cat_list):
    st.subheader("Bivariate Analysis")
    chart_type = st.selectbox("Choose chart type:", ["Scatter Plot", "Bar Plot", "Line Chart", "Correlation Coefficient"])
    if chart_type == "Scatter Plot":
        x = st.selectbox("Select Independent Variable (X, numerical):", num_list)
        y = st.selectbox("Select Dependent Variable (Y, numerical):", num_list)
        hue = st.selectbox("Optional Hue (categorical):", ["None"] + cat_list)
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue=None if hue == "None" else hue, data=df, ax=ax)
        ax.set_title(f"Scatter Plot: {y} vs {x}")
        st.pyplot(fig)
    elif chart_type == "Bar Plot":
        x = st.selectbox("Select Independent Variable (categorical):", cat_list)
        y = st.selectbox("Select Dependent Variable (numerical):", num_list)
        fig, ax = plt.subplots()
        sns.barplot(x=x, y=y, data=df, ax=ax)
        ax.set_title(f"Bar Plot: {y} grouped by {x}")
        st.pyplot(fig)
    elif chart_type == "Line Chart":
        x = st.selectbox("Select X-axis Variable (numerical or categorical):", df.columns)
        y = st.selectbox("Select Y-axis Variable (numerical):", num_list)
        fig, ax = plt.subplots()
        sns.lineplot(x=x, y=y, data=df, ax=ax)
        ax.set_title(f"Line Chart: {y} over {x}")
        st.pyplot(fig)
    elif chart_type == "Correlation Coefficient":
        x = st.selectbox("Select First Variable (numerical):", num_list)
        y = st.selectbox("Select Second Variable (numerical):", num_list)
        corr = df[x].corr(df[y])
        st.write(f"Correlation between {x} and {y}: {corr:.2f}")

#Multivariable Analysis
def multivariate_analysis(df, num_list, cat_list):
    st.subheader("Multivariate Analysis")
    
    # Allow user to select chart type
    chart_type = st.selectbox("Choose chart type:", 
                              ["Pair Plot", "Correlation Matrix", "Grouped Bar Chart", "Bubble Chart", "Heat Map"])
    
    # Pair Plot
    if chart_type == "Pair Plot":
        # Allow user to select numerical variables for the pair plot
        selected_vars = st.multiselect(
            "Select numerical variables to include in the pair plot:",
            options=num_list,
            default=num_list  # By default, select all numerical variables
        )
        
        # Optional Hue (categorical variable)
        hue = st.selectbox("Optional Hue (categorical):", ["None"] + cat_list)
        
        # Generate Pair Plot
        if not selected_vars:
            st.warning("Please select at least one variable for the pair plot.")
        else:
            pairplot_fig = sns.pairplot(df[selected_vars], hue=None if hue == "None" else hue)
            st.pyplot(pairplot_fig)

    # Correlation Matrix
    elif chart_type == "Correlation Matrix":
        selected_vars = st.multiselect(
            "Select numerical variables to include in the correlation matrix:",
            options=num_list,
            default=num_list  # By default, select all numerical variables
        )
        if not selected_vars:
            st.warning("Please select at least one variable.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[selected_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
    
    # Grouped Bar Chart
    elif chart_type == "Grouped Bar Chart":
        x = st.selectbox("Select X-axis Variable (categorical):", cat_list)
        hue = st.selectbox("Select Grouping Variable (categorical):", cat_list)
        fig, ax = plt.subplots()
        sns.countplot(x=x, hue=hue, data=df, ax=ax)
        ax.set_title(f"Grouped Bar Chart: {x} grouped by {hue}")
        st.pyplot(fig)

    # Bubble Chart
    elif chart_type == "Bubble Chart":
        x = st.selectbox("Select X-axis Variable (numerical):", num_list)
        y = st.selectbox("Select Y-axis Variable (numerical):", num_list)
        size = st.selectbox("Select Bubble Size Variable (numerical):", num_list)
        color = st.selectbox("Select Bubble Color Variable (categorical):", ["None"] + cat_list)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, size=size, hue=None if color == "None" else color, sizes=(20, 200), ax=ax)
        ax.set_title(f"Bubble Chart: {x} vs {y} (Size: {size})")
        st.pyplot(fig)
    
    # Heat Map
    elif chart_type == "Heat Map":
        x = st.selectbox("Select X-axis Variable (categorical):", cat_list)
        y = st.selectbox("Select Y-axis Variable (categorical):", cat_list)
        value = st.selectbox("Select Value Variable (numerical):", num_list)
        pivot_table = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
        fig, ax = plt.subplots()
        sns.heatmap(pivot_table, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title(f"Heat Map: {value} by {x} and {y}")
        st.pyplot(fig)
    
    # Heat Map
    elif chart_type == "Heat Map":
        x = st.selectbox("Select X-axis Variable (categorical):", cat_list)
        y = st.selectbox("Select Y-axis Variable (categorical):", cat_list)
        value = st.selectbox("Select Value Variable (numerical):", num_list)
        pivot_table = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
        fig, ax = plt.subplots()
        sns.heatmap(pivot_table, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title(f"Heat Map: {value} by {x} and {y}")
        st.pyplot(fig)

# Main App
st.title("Interactive EDA Application")
uploaded_file = st.file_uploader("Upload your dataset (CSV only):")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Convert date columns to datetime if detected
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                continue

    # Filter the dataset
    filtered_df = filter_data(df)

    st.write("### Filtered Dataset:")
    st.dataframe(filtered_df)

    num_list = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
    cat_list = [col for col in filtered_df.columns if pd.api.types.is_string_dtype(filtered_df[col])]

    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type:",
        ["Data Cleaning & Descriptive", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )
    
    if analysis_type == "Data Cleaning & Descriptive":
        data_cleaning_and_descriptive(df)
    elif analysis_type == "Univariate Analysis":
        univariate_analysis(filtered_df, num_list, cat_list)
    elif analysis_type == "Bivariate Analysis":
        bivariate_analysis(filtered_df, num_list, cat_list)
    elif analysis_type == "Multivariate Analysis":
        multivariate_analysis(filtered_df, num_list, cat_list)
