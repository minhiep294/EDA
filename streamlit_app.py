import io
import openpyxl
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


# Define save_chart_as_image function
def save_chart_as_image(fig, filename="chart.png"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer


# Helper function to validate DataFrame
def validate_dataframe(df):
    if df is None:
        st.error("The DataFrame is not defined.")
        return False
    if df.empty:
        st.error("The uploaded file resulted in an empty DataFrame. Please check the file.")
        return False
    return True


# Data Cleaning and Descriptive Statistics
def data_cleaning_and_descriptive(df):
    st.header("1. Data Cleaning")
    if not validate_dataframe(df):
        return

    st.subheader("Handle Missing Values")
    missing_option = st.radio(
        "Choose a method to handle missing values:",
        ("Leave as is", "Impute with Mean (Numerical Only)", "Remove Rows with Missing Data"),
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

    st.subheader("Remove Duplicates")
    if st.button("Remove Duplicate Rows"):
        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        st.write(f"Removed {before - after} duplicate rows.")

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


# Univariate Analysis
def univariate_analysis(df, num_list, cat_list):
    st.subheader("Univariate Analysis")
    variable_type = st.radio("Choose variable type:", ["Numerical", "Categorical"])

    if variable_type == "Numerical":
        col = st.selectbox("Select a numerical variable:", num_list)
        chart_type = st.selectbox(
            "Choose chart type:",
            ["Histogram", "Box Plot", "Density Plot", "QQ Plot"],
        )
        fig, ax = plt.subplots()

        if chart_type == "Histogram":
            sns.histplot(df[col], kde=True, ax=ax)
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

        st.pyplot(fig)

    elif variable_type == "Categorical":
        col = st.selectbox("Select a categorical variable:", cat_list)
        chart_type = st.selectbox("Choose chart type:", ["Count Plot", "Bar Chart"])
        fig, ax = plt.subplots()

        if chart_type == "Count Plot":
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Count Plot of {col}")
        elif chart_type == "Bar Chart":
            df[col].value_counts().plot.bar(ax=ax)
            ax.set_title(f"Bar Chart of {col}")

        st.pyplot(fig)
        
# Bivariate Analysis
def bivariate_analysis(df, num_list, cat_list):
    st.subheader("Bivariate Analysis")
    chart_type = st.selectbox("Choose chart type:", ["Scatter Plot", "Bar Plot", "Line Chart", "Correlation Coefficient"])
    if chart_type == "Scatter Plot":
        x = st.selectbox("Select Independent Variable (X, numerical):", num_list)
        y = st.selectbox("Select Dependent Variable (Y, numerical):", num_list)
        hue = st.selectbox("Optional Hue (categorical):", ["None"] + cat_list)
        sample_size = st.slider("Sample Size:", min_value=100, max_value=min(1000, len(df)), value=500)
        
        sampled_df = df.sample(n=sample_size, random_state=42)
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue=None if hue == "None" else hue, data=sampled_df, ax=ax)
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
        # Select numerical variables for the correlation matrix
        selected_vars = st.multiselect(
            "Select Variables for Correlation Analysis (default: all numerical):",
            num_list,
            default=num_list
        )
        # Ensure at least two variables are selected
        if len(selected_vars) < 2:
            st.warning("Please select at least two variables for correlation analysis.")
        else:
            advanced = st.checkbox("Advanced Options (Choose Correlation Method)")
            if advanced:
                corr_method = st.radio("Choose Correlation Method:", ["Pearson", "Spearman", "Kendall"])
            else:
                corr_method = "Pearson"  # Default method
    
            # Generate and display correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[selected_vars].corr(method=corr_method.lower())
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title(f"Correlation Matrix ({corr_method} Method)")
            st.pyplot(fig)
        # Add export option
        buffer = save_chart_as_image(fig)
        st.download_button(
            label="Download Chart as PNG",
            data=buffer,
            file_name=f"{col}_{chart_type.lower().replace(' ', '_')}.png",
            mime="image/png"
        )            
        
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
        # Add export option
        buffer = save_chart_as_image(fig)
        st.download_button(
            label="Download Chart as PNG",
            data=buffer,
            file_name=f"{col}_{chart_type.lower().replace(' ', '_')}.png",
            mime="image/png"
        )

def subgroup_analysis(df, num_list, cat_list):
    st.header("Subgroup Analysis")

    # Select Numerical and Categorical Variables
    st.markdown("### Select Variables for Subgroup Analysis")
    numerical_col = st.selectbox("Select Numerical Variable:", num_list)
    categorical_col = st.selectbox("Select Categorical Variable:", cat_list)

    # Allow user to choose the type of plot(s)
    st.markdown("### Choose the Chart Type(s)")
    chart_types = st.multiselect(
        "Select the chart type(s) you want to generate:",
        ["Box Plot", "Bar Chart"]
    )

    # Check if categorical_col is categorical, if not, convert it
    if not pd.api.types.is_categorical_dtype(df[categorical_col]):
        df[categorical_col] = df[categorical_col].astype("category")

    # Handle empty selection
    if not chart_types:
        st.warning("Please select at least one chart type to generate.")
        return

    # Generate Bar Chart if selected
    if "Bar Chart" in chart_types:
        st.markdown("#### Bar Chart")

        # Let user select the metric(s) to visualize
        selected_metrics = st.multiselect(
            "Choose the metric(s) for the bar chart:",
            ["Mean", "Sum", "Standard Deviation"]
        )

        # Handle case where no metric is selected
        if not selected_metrics:
            st.warning("Please select at least one metric for the bar chart.")
            return

        # Calculate requested statistics
        metrics_map = {
            "Mean": "mean",
            "Sum": "sum",
            "Standard Deviation": "std"
        }
        selected_agg_funcs = [metrics_map[metric] for metric in selected_metrics]
        grouped = df.groupby(categorical_col)[numerical_col].agg(selected_agg_funcs).reset_index()

        # Generate bar charts for each selected metric
        for metric in selected_metrics:
            agg_func = metrics_map[metric]
            fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
            sns.barplot(data=grouped, x=categorical_col, y=agg_func, ax=ax_bar)
            ax_bar.set_title(f"{metric} of {numerical_col} by {categorical_col}")
            ax_bar.set_xlabel(categorical_col)
            ax_bar.set_ylabel(metric)
            st.pyplot(fig_bar)

            # Add download button for each bar chart
            buffer_bar = save_chart_as_image(fig_bar)
            st.download_button(
                label=f"Download {metric} Bar Chart as PNG",
                data=buffer_bar,
                file_name=f"bar_chart_{agg_func}_{numerical_col}_by_{categorical_col}.png",
                mime="image/png",
            )

    # Generate Box Plot if selected
    if "Box Plot" in chart_types:
        st.markdown("#### Box Plot")
        fig_box, ax_box = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x=categorical_col, y=numerical_col, ax=ax_box)
        ax_box.set_title(f"Box Plot of {numerical_col} by {categorical_col}")
        ax_box.set_xlabel(categorical_col)
        ax_box.set_ylabel(numerical_col)
        st.pyplot(fig_box)

        # Add download button for box plot
        buffer_box = save_chart_as_image(fig_box)
        st.download_button(
            label="Download Box Plot as PNG",
            data=buffer_box,
            file_name=f"box_plot_{numerical_col}_by_{categorical_col}.png",
            mime="image/png",
        )

# Helper function to save charts as images
def save_chart_as_image(fig):
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer
    
# Linear Regression Section
def linear_regression_analysis(df, num_list, cat_list):
    st.subheader("Linear Regression Analysis")

    regression_type = st.radio("Choose Regression Type:", ["Simple Regression", "Multiple Regression"])

    if regression_type == "Simple Regression":
        x_col = st.selectbox("Select Independent Variable (X):", num_list + cat_list)
        y_col = st.selectbox("Select Dependent Variable (Y):", num_list)

        if x_col and y_col:
            try:
                if x_col in cat_list:
                    X = pd.get_dummies(df[x_col], drop_first=True)
                else:
                    X = df[[x_col]]
                y = df[y_col]

                # Combine and drop missing values
                combined_data = pd.concat([X, y], axis=1).dropna()
                X = combined_data.iloc[:, :-1]
                y = combined_data.iloc[:, -1]

                # Add constant for intercept
                X = sm.add_constant(X)

                # Fit the model
                model = sm.OLS(y, X).fit()

                # Display results
                st.markdown("### Regression Results Summary")
                st.text(model.summary())

                # Residuals plot
                residuals = model.resid
                fitted_vals = model.fittedvalues
                fig, ax = plt.subplots()
                sns.residplot(x=fitted_vals, y=residuals, lowess=True, ax=ax, line_kws={"color": "red", "lw": 1})
                ax.set_title("Residuals vs Fitted")
                ax.set_xlabel("Fitted Values")
                ax.set_ylabel("Residuals")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred during Simple Linear Regression: {e}")

    elif regression_type == "Multiple Regression":
        x_cols = st.multiselect("Select Independent Variables (X):", num_list + cat_list)
        y_col = st.selectbox("Select Dependent Variable (Y):", num_list)

        if x_cols and y_col:
            try:
                X = pd.get_dummies(df[x_cols], drop_first=True)
                y = df[y_col]

                # Combine and drop missing values
                combined_data = pd.concat([X, y], axis=1).dropna()
                X = combined_data.iloc[:, :-1]
                y = combined_data.iloc[:, -1]

                # Add constant for intercept
                X = sm.add_constant(X)

                # Fit the model
                model = sm.OLS(y, X).fit()

                # Display results
                st.markdown("### Regression Results Summary")
                st.text(model.summary())

                # Residuals plot
                residuals = model.resid
                fitted_vals = model.fittedvalues
                fig, ax = plt.subplots()
                sns.residplot(x=fitted_vals, y=residuals, lowess=True, ax=ax, line_kws={"color": "red", "lw": 1})
                ax.set_title("Residuals vs Fitted")
                ax.set_xlabel("Fitted Values")
                ax.set_ylabel("Residuals")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred during Multiple Linear Regression: {e}")
                
# Main Functionality
def main():
    st.title("Interactive EDA and Regression App")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()

            # Handle file types
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return

            if df.empty:
                st.error("The uploaded file is empty. Please check the file.")
                return

            # Dataset preview
            st.write("### Dataset Preview:")
            st.dataframe(df.head())

            # Identify numerical and categorical columns
            num_list = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            cat_list = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

            # Sidebar navigation
            st.sidebar.title("Navigation")
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

            # Navigate based on analysis type
            if analysis_type == "Data Cleaning & Descriptive":
                data_cleaning_and_descriptive(df)
            elif analysis_type == "Univariate Analysis":
                univariate_analysis(df, num_list, cat_list)
            elif analysis_type == "Bivariate Analysis":
                bivariate_analysis(df, num_list, cat_list)
            elif analysis_type == "Multivariate Analysis":
                multivariate_analysis(df, num_list)
            elif analysis_type == "Linear Regression":
                linear_regression_analysis(df, num_list)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a file to proceed.")


if __name__ == "__main__":
    main()
