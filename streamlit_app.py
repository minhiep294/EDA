import io
import base64
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm
import google.generativeai as genai
import os

# Define save_chart_as_image function
def save_chart_as_image(fig, filename="chart.png"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer

# Set up Gemini API Key
GEMINI_API_KEY = st.secrets["gemini_api_key"] if "gemini_api_key" in st.secrets else st.text_input("Enter Google Gemini API Key:", type="password")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Function to encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to describe chart using Gemini API
def describe_chart_with_gemini(image_path):
    if not GEMINI_API_KEY:
        st.error("Please enter a valid Google Gemini API key.")
        return "No API Key provided."

    try:
        # Load Gemini model
        model = genai.GenerativeModel("gemini-pro-vision")

        # Open the image and generate the description
        with open(image_path, "rb") as image_file:
            response = model.generate_content(
                ["Describe this chart and summarize its insights."],
                image=image_file
            )

        # Extract the text from the response
        description = response.text
        return description
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return "Failed to generate AI description."

# Function to save chart as an image
def save_chart_as_image(fig, filename="chart.png"):
    fig.savefig(filename, bbox_inches="tight")
    return filename
        
# Section: Data Cleaning and Descriptive Statistics
def data_cleaning_and_descriptive(df):
    st.header("1. Data Cleaning")

    # Handle Missing Values
    st.subheader("Handle Missing Values")
    missing_option = st.radio(
        "Choose a method to handle missing values:",
        ("Leave as is", "Impute with Mean (Numerical Only)", "Remove Rows with Missing Data")
    )
    if missing_option == "Impute with Mean (Numerical Only)":
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

def univariate_analysis(df, num_list, cat_list):
    st.subheader("Univariate Analysis")
    variable_type = st.radio("Choose variable type:", ["Numerical", "Categorical"])

    # Checkbox for AI Description
    use_ai_description = st.checkbox("Use AI to describe the chart")

    # Numerical Analysis
    if variable_type == "Numerical":
        col = st.selectbox("Select a numerical variable:", num_list)
        chart_type = st.selectbox("Choose chart type:", ["Histogram", "Box Plot", "Density Plot", "QQ Plot"])

        # Generate the chart
        fig, ax = plt.subplots()
        if chart_type == "Histogram":
            bins = st.slider("Number of bins:", min_value=5, max_value=50, value=20)
            sns.histplot(df[col], bins=bins, kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
        elif chart_type == "Box Plot":
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Box Plot of {col}")
        elif chart_type == "Density Plot":
            sns.kdeplot(df[col], fill=True, ax=ax)
            ax.set_title(f"Density Plot of {col}")
        elif chart_type == "QQ Plot":
            stats.probplot(df[col], dist="norm", plot=ax)
            ax.set_title(f"QQ Plot of {col}")
        
        # Save and display the chart
        chart_path = save_chart_as_image(fig, filename=f"univariate_{col}_{chart_type}.png")
        st.pyplot(fig)

        # Generate AI description if enabled
        if use_ai_description:
            with st.spinner("Generating AI description for the chart..."):
                ai_description = describe_chart_with_gemini(chart_path)
            st.markdown(f"**AI Description:** {ai_description}")

    # Categorical Analysis
    elif variable_type == "Categorical":
        col = st.selectbox("Select a categorical variable:", cat_list)
        chart_type = st.selectbox("Choose chart type:", ["Count Plot", "Bar Chart", "Pie Chart"])

        # Generate the chart
        fig, ax = plt.subplots()
        if chart_type == "Count Plot":
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Count Plot of {col}")
        elif chart_type == "Bar Chart":
            df[col].value_counts().plot.bar(ax=ax)
            ax.set_title(f"Bar Chart of {col}")
        elif chart_type == "Pie Chart":
            df[col].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {col}")

        # Save and display the chart
        chart_path = save_chart_as_image(fig, filename=f"univariate_{col}_{chart_type}.png")
        st.pyplot(fig)

        # Generate AI description if enabled
        if use_ai_description:
            with st.spinner("Generating AI description for the chart..."):
                ai_description = describe_chart_with_gemini(chart_path)
            st.markdown(f"**AI Description:** {ai_description}")

# Function to Save Chart as Image
def save_chart_as_image(fig, filename="chart.png"):
    """Saves the chart as an image file and returns the file path."""
    fig.savefig(filename, bbox_inches="tight")
    return filename

# Function to Describe Chart with Gemini API
def describe_chart_with_gemini(image_path):
    """Sends a chart image to Gemini API for description."""
    try:
        import google.generativeai as genai
        from google.generativeai.types import content_types

        # Configure Gemini API Key
        if "gemini_api_key" in st.secrets:
            genai.configure(api_key=st.secrets["gemini_api_key"])
        else:
            st.error("Please provide your Google Gemini API key.")
            return "No API Key provided."

        # Load Gemini model
        model = genai.GenerativeModel("gemini-pro-vision")

        # Generate description
        with open(image_path, "rb") as image_file:
            response = model.generate_content(
                ["Describe this chart and summarize its insights."],
                image=image_file
            )
        return response.text

    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return "Failed to generate AI description."
        
# Bivariate Analysis
def bivariate_analysis(df, num_list, cat_list):
    st.subheader("Bivariate Analysis")

    # Chart Type Selection
    chart_type = st.selectbox(
        "Choose chart type:", 
        ["Scatter Plot", "Bar Plot", "Line Chart", "Correlation Coefficient", "Subgroup Analysis"]
    )

    use_ai_description = st.checkbox("Use AI to describe the chart")  # Optional AI description

    # Scatter Plot
    if chart_type == "Scatter Plot":
        x = st.selectbox("Select X-axis (numerical):", num_list)
        y = st.selectbox("Select Y-axis (numerical):", num_list)
        hue = st.selectbox("Optional Hue (categorical):", ["None"] + cat_list)
        sample_size = st.slider("Sample Size:", min_value=100, max_value=min(1000, len(df)), value=500)

        sampled_df = df.sample(n=sample_size, random_state=42)
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue=None if hue == "None" else hue, data=sampled_df, ax=ax)
        ax.set_title(f"Scatter Plot: {y} vs {x}")
        chart_path = save_chart_as_image(fig, filename="scatter_plot.png")
        st.pyplot(fig)

        if use_ai_description:
            with st.spinner("Generating AI description for the chart..."):
                ai_description = describe_chart_with_gemini(chart_path)
            st.markdown(f"**AI Description:** {ai_description}")

    # Bar Plot
    elif chart_type == "Bar Plot":
        x = st.selectbox("Select X-axis (categorical):", cat_list)
        y = st.selectbox("Select Y-axis (numerical):", num_list)
        fig, ax = plt.subplots()
        sns.barplot(x=x, y=y, data=df, ci=None, ax=ax)
        ax.set_title(f"Bar Plot: {y} grouped by {x}")
        chart_path = save_chart_as_image(fig, filename="bar_plot.png")
        st.pyplot(fig)

        if use_ai_description:
            with st.spinner("Generating AI description for the chart..."):
                ai_description = describe_chart_with_gemini(chart_path)
            st.markdown(f"**AI Description:** {ai_description}")

    # Line Chart
    elif chart_type == "Line Chart":
        x = st.selectbox("Select X-axis Variable:", df.columns)
        y = st.selectbox("Select Y-axis Variable (numerical):", num_list)
        fig, ax = plt.subplots()
        sns.lineplot(x=x, y=y, data=df, ax=ax)
        ax.set_title(f"Line Chart: {y} over {x}")
        chart_path = save_chart_as_image(fig, filename="line_chart.png")
        st.pyplot(fig)

        if use_ai_description:
            with st.spinner("Generating AI description for the chart..."):
                ai_description = describe_chart_with_gemini(chart_path)
            st.markdown(f"**AI Description:** {ai_description}")

    # Correlation Matrix
    elif chart_type == "Correlation Coefficient":
        selected_vars = st.multiselect("Select numerical variables:", num_list, default=num_list)
        if len(selected_vars) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[selected_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            chart_path = save_chart_as_image(fig, filename="correlation_matrix.png")
            st.pyplot(fig)

            if use_ai_description:
                with st.spinner("Generating AI description for the chart..."):
                    ai_description = describe_chart_with_gemini(chart_path)
                st.markdown(f"**AI Description:** {ai_description}")
        else:
            st.warning("Please select at least two numerical variables.")

    # Subgroup Analysis
    elif chart_type == "Subgroup Analysis":
        st.subheader("Subgroup Analysis")

        num_col = st.selectbox("Select Numerical Variable:", num_list)
        cat_col = st.selectbox("Select Categorical Variable:", cat_list)
        chart_types = st.multiselect("Select Charts to Generate:", ["Box Plot", "Bar Chart", "Pie Chart"])

        if "Box Plot" in chart_types:
            try:
                st.markdown("### Box Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
                chart_path = save_chart_as_image(fig, filename="subgroup_box_plot.png")
                st.pyplot(fig)

                if use_ai_description:
                    with st.spinner("Generating AI description for the chart..."):
                        ai_description = describe_chart_with_gemini(chart_path)
                    st.markdown(f"**AI Description:** {ai_description}")
            except Exception as e:
                st.error(f"Error generating Box Plot: {e}")

        if "Bar Chart" in chart_types:
            st.markdown("### Bar Chart")
            agg_funcs = st.multiselect("Select Metrics for Bar Chart:", ["mean", "sum", "count"], default=["mean"])
            try:
                grouped = df.groupby(cat_col)[num_col].agg(agg_funcs).reset_index()
                for func in agg_funcs:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=cat_col, y=func, data=grouped, ci=None, ax=ax)
                    ax.set_title(f"{func.capitalize()} of {num_col} by {cat_col}")
                    chart_path = save_chart_as_image(fig, filename=f"subgroup_bar_chart_{func}.png")
                    st.pyplot(fig)

                    if use_ai_description:
                        with st.spinner("Generating AI description for the chart..."):
                            ai_description = describe_chart_with_gemini(chart_path)
                        st.markdown(f"**AI Description:** {ai_description}")
            except Exception as e:
                st.error(f"Error generating Bar Chart: {e}")

        if "Pie Chart" in chart_types:
            st.markdown("### Pie Chart")
            agg_func = st.selectbox("Select Aggregation for Pie Chart:", ["mean", "sum", "count"], index=0)
            try:
                grouped = df.groupby(cat_col)[num_col].agg(agg_func).reset_index()
                grouped = grouped.sort_values(by=num_col, ascending=False)
                fig, ax = plt.subplots()
                ax.pie(grouped[num_col], labels=grouped[cat_col], autopct="%1.1f%%", startangle=90)
                ax.set_title(f"{agg_func.capitalize()} of {num_col} by {cat_col}")
                chart_path = save_chart_as_image(fig, filename="subgroup_pie_chart.png")
                st.pyplot(fig)

                if use_ai_description:
                    with st.spinner("Generating AI description for the chart..."):
                        ai_description = describe_chart_with_gemini(chart_path)
                    st.markdown(f"**AI Description:** {ai_description}")
            except Exception as e:
                st.error(f"Error generating Pie Chart: {e}")
        
def multivariate_analysis(df, num_list, cat_list):
    st.subheader("Multivariate Analysis")

    chart_type = st.selectbox("Choose chart type:", 
                              ["Pair Plot", "Correlation Matrix", "Grouped Bar Chart", "Bubble Chart", "Heat Map"])

    use_ai_description = st.checkbox("Use AI to describe the chart")  # Toggle AI chart descriptions

    # Pair Plot
    if chart_type == "Pair Plot":
        selected_vars = st.multiselect("Select numerical variables for Pair Plot:", num_list, default=num_list)
        hue = st.selectbox("Optional Hue (categorical):", ["None"] + cat_list)
        if selected_vars:
            try:
                pairplot_fig = sns.pairplot(df[selected_vars], hue=None if hue == "None" else hue)
                chart_path = save_chart_as_image(pairplot_fig.fig, filename="pair_plot.png")
                st.pyplot(pairplot_fig)

                # AI description if enabled
                if use_ai_description:
                    with st.spinner("Generating AI description for the chart..."):
                        ai_description = describe_chart_with_gemini(chart_path)
                    st.markdown(f"**AI Description:** {ai_description}")
            except Exception as e:
                st.error(f"Error generating Pair Plot: {e}")

    # Correlation Matrix
    elif chart_type == "Correlation Matrix":
        selected_vars = st.multiselect("Select numerical variables:", num_list, default=num_list)
        if len(selected_vars) >= 2:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = df[selected_vars].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Matrix")
                chart_path = save_chart_as_image(fig, filename="correlation_matrix.png")
                st.pyplot(fig)

                # AI description if enabled
                if use_ai_description:
                    with st.spinner("Generating AI description for the chart..."):
                        ai_description = describe_chart_with_gemini(chart_path)
                    st.markdown(f"**AI Description:** {ai_description}")
            except Exception as e:
                st.error(f"Error generating Correlation Matrix: {e}")
        else:
            st.warning("Select at least two numerical variables for correlation analysis.")

    # Grouped Bar Chart
    elif chart_type == "Grouped Bar Chart":
        x = st.selectbox("Select X-axis (categorical):", cat_list)
        hue = st.selectbox("Select Grouping Variable (categorical):", cat_list)
        try:
            fig, ax = plt.subplots()
            sns.countplot(x=x, hue=hue, data=df, ax=ax)
            ax.set_title(f"Grouped Bar Chart: {x} grouped by {hue}")
            chart_path = save_chart_as_image(fig, filename="grouped_bar_chart.png")
            st.pyplot(fig)

            # AI description if enabled
            if use_ai_description:
                with st.spinner("Generating AI description for the chart..."):
                    ai_description = describe_chart_with_gemini(chart_path)
                st.markdown(f"**AI Description:** {ai_description}")
        except Exception as e:
            st.error(f"Error generating Grouped Bar Chart: {e}")

    # Bubble Chart
    elif chart_type == "Bubble Chart":
        x = st.selectbox("Select X-axis (numerical):", num_list)
        y = st.selectbox("Select Y-axis (numerical):", num_list)
        size = st.selectbox("Select Bubble Size (numerical):", num_list)
        color = st.selectbox("Select Bubble Color (categorical):", ["None"] + cat_list)
        try:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x, y=y, size=size, hue=None if color == "None" else color, sizes=(20, 200), ax=ax)
            ax.set_title(f"Bubble Chart: {y} vs {x}")
            chart_path = save_chart_as_image(fig, filename="bubble_chart.png")
            st.pyplot(fig)

            # AI description if enabled
            if use_ai_description:
                with st.spinner("Generating AI description for the chart..."):
                    ai_description = describe_chart_with_gemini(chart_path)
                st.markdown(f"**AI Description:** {ai_description}")
        except Exception as e:
            st.error(f"Error generating Bubble Chart: {e}")

    # Heat Map
    elif chart_type == "Heat Map":
        x = st.selectbox("Select X-axis (categorical):", cat_list)
        y = st.selectbox("Select Y-axis (categorical):", cat_list)
        value = st.selectbox("Select Value (numerical):", num_list)
        try:
            pivot_table = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
            fig, ax = plt.subplots()
            sns.heatmap(pivot_table, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title(f"Heat Map: {value} by {x} and {y}")
            chart_path = save_chart_as_image(fig, filename="heat_map.png")
            st.pyplot(fig)

            # AI description if enabled
            if use_ai_description:
                with st.spinner("Generating AI description for the chart..."):
                    ai_description = describe_chart_with_gemini(chart_path)
                st.markdown(f"**AI Description:** {ai_description}")
        except Exception as e:
            st.error(f"Error generating Heat Map: {e}")

# Function to Save Chart as Image
def save_chart_as_image(fig, filename="chart.png"):
    """Saves the chart as an image file and returns the file path."""
    fig.savefig(filename, bbox_inches="tight")
    return filename

# Function to Describe Chart with Gemini API
def describe_chart_with_gemini(image_path):
    """Sends a chart image to Gemini API for description."""
    try:
        import google.generativeai as genai
        if "gemini_api_key" in st.secrets:
            genai.configure(api_key=st.secrets["gemini_api_key"])
        else:
            st.error("Please provide your Google Gemini API key.")
            return "No API Key provided."

        model = genai.GenerativeModel("gemini-pro-vision")
        with open(image_path, "rb") as image_file:
            response = model.generate_content(["Describe this chart and summarize its insights."], image=image_file)
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return "Failed to generate AI description."

# Linear Regression Analysis
def clean_and_prepare_data(df, x_columns, y_column, cat_list):
    """
    Cleans and validates input data for regression analysis.
    Converts categorical variables into dummies and ensures all data is numeric.
    """
    try:
        if not x_columns or not y_column:
            return None, None  # Return None if no variables are selected

        # Select relevant columns
        X = df[x_columns].copy()

        # Convert categorical columns to dummy variables
        for col in x_columns:
            if col in cat_list:
                X = pd.get_dummies(X, columns=[col], drop_first=True, dtype=float)

        # Ensure all X columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce')

        # Process the dependent variable (y)
        y = pd.to_numeric(df[y_column], errors='coerce')

        # Drop rows with NaN values
        combined = pd.concat([X, y], axis=1).dropna()
        X_clean = combined.drop(columns=[y_column])
        y_clean = combined[y_column]

        return X_clean, y_clean

    except Exception as e:
        st.error(f"Error during data cleaning: {e}")
        return None, None

def linear_regression_analysis(df, num_list, cat_list):
    """
    Main function for performing linear regression analysis with optional AI explanations.
    """
    st.subheader("Linear Regression Analysis")

    # Allow users to enable AI explanations
    use_ai_description = st.checkbox("Use AI to describe the regression results")

    # Choose regression type
    regression_type = st.radio("Select Regression Type:", ["Simple Regression", "Multiple Regression"])

    # Simple Linear Regression
    if regression_type == "Simple Regression":
        x = st.selectbox("Select Independent Variable (X):", ["Select Variable"] + num_list + cat_list)
        y = st.selectbox("Select Dependent Variable (Y):", ["Select Variable"] + num_list)

        if x != "Select Variable" and y != "Select Variable":
            try:
                # Clean and prepare data
                X, y_values = clean_and_prepare_data(df, [x], y, cat_list)

                if X is not None and y_values is not None:
                    # Add a constant for the intercept
                    X = sm.add_constant(X)

                    # Fit the regression model
                    model = sm.OLS(y_values, X).fit()
                    st.write(model.summary())

                    # Generate AI explanation if enabled
                    if use_ai_description:
                        with st.spinner("Generating AI description for the regression results..."):
                            ai_description = describe_regression_with_gemini(model.summary().as_text())
                        st.markdown("### AI Description:")
                        st.markdown(f"**{ai_description}**")

            except Exception as e:
                st.error(f"Error during regression analysis: {e}")

    # Multiple Linear Regression
    elif regression_type == "Multiple Regression":
        x_cols = st.multiselect("Select Independent Variables (X):", num_list + cat_list)
        y = st.selectbox("Select Dependent Variable (Y):", ["Select Variable"] + num_list)

        if x_cols and y != "Select Variable":
            try:
                # Clean and prepare data
                X, y_values = clean_and_prepare_data(df, x_cols, y, cat_list)

                if X is not None and y_values is not None:
                    # Add a constant for the intercept
                    X = sm.add_constant(X)

                    # Fit the regression model
                    model = sm.OLS(y_values, X).fit()
                    st.write(model.summary())

                    # Generate AI explanation if enabled
                    if use_ai_description:
                        with st.spinner("Generating AI description for the regression results..."):
                            ai_description = describe_regression_with_gemini(model.summary().as_text())
                        st.markdown("### AI Description:")
                        st.markdown(f"**{ai_description}**")

            except Exception as e:
                st.error(f"Error during regression analysis: {e}")


def clean_and_prepare_data(df, x_cols, y_col, cat_list):
    """
    Cleans and prepares the data for regression analysis.
    Converts categorical variables into dummy variables if needed.
    """
    try:
        # Prepare X (independent variables)
        X = df[x_cols]
        for col in x_cols:
            if col in cat_list:
                X = pd.get_dummies(X, columns=[col], drop_first=True)

        # Prepare y (dependent variable)
        y = df[y_col]

        # Drop missing values
        combined_data = pd.concat([X, y], axis=1).dropna()
        X_cleaned = combined_data.iloc[:, :-1]
        y_cleaned = combined_data.iloc[:, -1]

        return X_cleaned, y_cleaned
    except Exception as e:
        st.error(f"Error cleaning and preparing data: {e}")
        return None, None


def describe_regression_with_gemini(regression_summary):
    """
    Sends the regression summary to Gemini API and gets a descriptive explanation.
    """
    try:
        import google.generativeai as genai

        # Configure Gemini API Key
        if "gemini_api_key" in st.secrets:
            genai.configure(api_key=st.secrets["gemini_api_key"])
        else:
            st.error("Please provide your Google Gemini API key.")
            return "No API Key provided."

        # Send regression summary to Gemini
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(
            [f"Provide a detailed, simple-to-understand explanation for the following regression summary:\n\n{regression_summary}"]
        )
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return "Failed to generate AI description."


                
# Main App
st.title("Interactive EDA App")
# Initialize df to None
df = None

# File Upload Section
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    try:
        # Load CSV or Excel file based on the extension
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            sheet_names = pd.ExcelFile(uploaded_file, engine="openpyxl").sheet_names
            selected_sheet = st.selectbox("Select sheet to load", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, engine="openpyxl")
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            df = None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None

# Proceed only if df is successfully loaded
if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Sidebar Navigation
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type:",
        [
            "Data Cleaning & Descriptive",
            "Univariate Analysis",
            "Bivariate Analysis",
            "Multivariate Analysis",
            "Linear Regression",
        ],
    )

    # Identify Numerical and Categorical Columns
    num_list = df.select_dtypes(include=np.number).columns.tolist()
    cat_list = df.select_dtypes(include="object").columns.tolist()

    if not num_list:
        st.warning("No numerical columns found in the dataset.")
    if not cat_list:
        st.warning("No categorical columns found in the dataset.")

    # Perform Analysis Based on Selection
    if analysis_type == "Data Cleaning & Descriptive":
        data_cleaning_and_descriptive(df)
    elif analysis_type == "Univariate Analysis":
        univariate_analysis(df, num_list, cat_list)
    elif analysis_type == "Bivariate Analysis":
        bivariate_analysis(df, num_list, cat_list)
    elif analysis_type == "Multivariate Analysis":
        filtered_df = filter_data(df)  # Apply filters
        multivariate_analysis(filtered_df, num_list, cat_list)
    elif analysis_type == "Linear Regression":
        filtered_df = filter_data(df)  # Apply filters
        linear_regression_analysis(filtered_df, num_list, cat_list)

else:
    st.warning("Please upload a dataset to begin.")

