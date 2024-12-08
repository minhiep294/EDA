import io
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm

# Define save_chart_as_image function
def save_chart_as_image(fig, filename="chart.png"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer

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
    
    #Numerical
    if variable_type == "Numerical":
        col = st.selectbox("Select a numerical variable:", num_list)
        chart_type = st.selectbox(
            "Choose chart type:",
            ["Histogram", "Box Plot", "Density Plot", "QQ Plot"]
        )
        fig, ax = plt.subplots()
        
        # Statistics for numerical columns
        mean = df[col].mean()
        median = df[col].median()
        std_dev = df[col].std()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        num_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

        if chart_type == "Histogram":
            bins = st.slider("Number of bins:", min_value=5, max_value=50, value=20)
            log_scale = st.checkbox("Log Scale (X-axis)")
            sns.histplot(df[col], bins=bins, kde=True, ax=ax)
            if log_scale:
                ax.set_xscale("log")
            ax.set_title(f"Histogram of {col}")
        elif chart_type == "Box Plot":
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Box Plot of {col} (Outliers: {num_outliers})")
        elif chart_type == "Density Plot":
            sns.kdeplot(df[col], fill=True, ax=ax)
            ax.set_title(f"Density Plot of {col}")
        elif chart_type == "QQ Plot":
            stats.probplot(df[col], dist="norm", plot=ax)
            ax.set_title(f"QQ Plot of {col}")
        
        st.pyplot(fig)
        st.write(
            f"**Statistics for {col}:**\n"
            f"- Mean: {mean:.2f}\n"
            f"- Median: {median:.2f}\n"
            f"- Standard Deviation: {std_dev:.2f}\n"
            f"- Outliers (IQR method): {num_outliers}"
        )

    #Categorical
    elif variable_type == "Categorical":
        col = st.selectbox("Select a categorical variable:", cat_list)
        chart_type = st.selectbox(
            "Choose chart type:",
            ["Count Plot", "Bar Chart", "Pie Chart", "Box Plot"]
        )
        fig, ax = plt.subplots()
        
        if chart_type == "Count Plot":
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Count Plot of {col}")
        elif chart_type == "Bar Chart":
            df[col].value_counts().plot.bar(ax=ax)
            ax.set_title(f"Bar Chart of {col}")
        elif chart_type == "Pie Chart":
            df[col].value_counts().plot.pie(
                autopct="%1.1f%%", startangle=90, ax=ax
            )
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {col}")
        elif chart_type == "Box Plot":
            num_col = st.selectbox("Select a numerical variable for Box Plot:", num_list)
            sns.boxplot(x=col, y=num_col, data=df, ax=ax)
            ax.set_title(f"Box Plot of {num_col} by {col}")
        
        st.pyplot(fig)
        st.write(f"**Category Counts:**\n{df[col].value_counts()}")
        # Add export option
        buffer = save_chart_as_image(fig)
        st.download_button(
            label="Download Chart as PNG",
            data=buffer,
            file_name=f"{col}_{chart_type.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
        
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

    # Choose between Simple and Multiple Linear Regression
    regression_type = st.radio("Choose Regression Type:", ["Simple Regression", "Multiple Regression"])

    if regression_type == "Simple Regression":
        st.markdown("### Simple Linear Regression")
        x_col = st.selectbox("Select Independent Variable (X):", num_list + cat_list)
        y_col = st.selectbox("Select Dependent Variable (Y):", num_list)

        if x_col and y_col:
            try:
                # Prepare data
                if x_col in cat_list:  # If X is categorical, convert to dummy variables
                    X = pd.get_dummies(df[x_col], drop_first=True)
                else:
                    X = df[[x_col]]  # Independent variable
                
                y = df[y_col]  # Dependent variable
                
                # Combine and drop missing values
                combined_data = pd.concat([X, y], axis=1).dropna()
                X = combined_data.iloc[:, :-1]
                y = combined_data.iloc[:, -1]

                # Ensure data contains no invalid values
                if not np.isfinite(X.values).all() or not np.isfinite(y.values).all():
                    st.error("The data contains invalid values (e.g., NaN or Infinity). Please clean your dataset.")
                    return

                # Fit the model
                model = LinearRegression()
                model.fit(X, y)

                # Get predictions and metrics
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                intercept = model.intercept_

                # Prepare coefficients table
                coef_df = pd.DataFrame({
                    "Variable": ["Intercept"] + list(X.columns),
                    "Coefficient": [intercept] + list(model.coef_)
                })
                st.markdown("### Regression Results")
                st.markdown("#### Model Coefficients")
                st.table(coef_df)

                # Model metrics
                results_df = pd.DataFrame({
                    "Metric": ["R-squared"],
                    "Value": [r2]
                })
                st.markdown("#### Model Metrics")
                st.table(results_df)

                # Residuals plot
                residuals = y - y_pred
                fig, ax = plt.subplots()
                sns.residplot(x=y_pred, y=residuals, lowess=True, ax=ax, line_kws={"color": "red", "lw": 1})
                ax.set_title("Residuals Plot")
                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Residuals")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred during Simple Linear Regression: {e}")

    elif regression_type == "Multiple Regression":
        st.markdown("### Multiple Linear Regression")
        x_cols = st.multiselect("Select Independent Variables (X):", num_list + cat_list)
        y_col = st.selectbox("Select Dependent Variable (Y):", num_list)

        if x_cols and y_col:
            try:
                # Prepare data
                X = df[x_cols]

                # Convert categorical variables to dummy variables
                X = pd.get_dummies(X, drop_first=True)
                y = df[y_col]

                # Combine and drop missing values
                combined_data = pd.concat([X, y], axis=1).dropna()
                X = combined_data.iloc[:, :-1]
                y = combined_data.iloc[:, -1]

                # Ensure data contains no invalid values
                if not np.isfinite(X.values).all() or not np.isfinite(y.values).all():
                    st.error("The data contains invalid values (e.g., NaN or Infinity). Please clean your dataset.")
                    return

                # Fit the model
                model = LinearRegression()
                model.fit(X, y)

                # Get predictions and metrics
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

                # Prepare coefficients table
                coef_df = pd.DataFrame({
                    "Variable": ["Intercept"] + list(X.columns),
                    "Coefficient": [model.intercept_] + list(model.coef_)
                })
                st.markdown("### Regression Results")
                st.markdown("#### Model Coefficients")
                st.table(coef_df)

                # Model metrics
                metrics_df = pd.DataFrame({
                    "Metric": ["R-squared", "Adjusted R-squared", "Mean Squared Error (MSE)"],
                    "Value": [r2, adj_r2, mse]
                })
                st.markdown("#### Model Metrics")
                st.table(metrics_df)

                # Residuals plot
                residuals = y - y_pred
                fig, ax = plt.subplots()
                sns.residplot(x=y_pred, y=residuals, lowess=True, ax=ax, line_kws={"color": "red", "lw": 1})
                ax.set_title("Residuals Plot")
                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Residuals")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred during Multiple Linear Regression: {e}")
                
# Main App
# File Upload Section
st.title("Interactive EDA Application")
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel only):", type=["csv", "xlsx"])

if uploaded_file:
    # Determine the file type based on its extension
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    try:
        if file_extension == "csv":
            # Read CSV file
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            # Read Excel file
            sheet_name = st.text_input("Enter sheet name (leave blank for default):", "")
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name if sheet_name else None)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            df = None
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        df = None
    
    if df is not None:
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
            ["Data Cleaning & Descriptive", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Subgroup Analysis", "Linear Regression"]
        )

        if analysis_type == "Data Cleaning & Descriptive":
            data_cleaning_and_descriptive(df)
        elif analysis_type == "Univariate Analysis":
            univariate_analysis(filtered_df, num_list, cat_list)
        elif analysis_type == "Bivariate Analysis":
            bivariate_analysis(filtered_df, num_list, cat_list)
        elif analysis_type == "Multivariate Analysis":
            multivariate_analysis(filtered_df, num_list, cat_list)
        elif analysis_type == "Linear Regression":
            linear_regression_analysis(filtered_df, num_list, cat_list)
        elif analysis_type == "Subgroup Analysis":
            subgroup_analysis(filtered_df, num_list, cat_list)
