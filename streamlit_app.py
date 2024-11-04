import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Purpose of Analysis
st.title("📊 General EDA Dashboard")

# Load dataset uploaded by the user
uploaded_data_file = st.file_uploader("Upload a dataset (.csv)", type=["csv"])

if uploaded_data_file:
    # Read the dataset
    data = pd.read_csv(uploaded_data_file)

    # Display raw data preview
    st.subheader("Raw Data Preview")
    st.write(data.head())  # Show first few rows of the uploaded dataset

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write("The table below provides summary statistics for all numeric columns.")
    st.write(data.describe())

    # Sidebar filters for numeric and categorical columns
    st.sidebar.title("Filter Options")
    
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Filtering by categorical column
    selected_category_column = st.sidebar.selectbox("Select Categorical Column to Filter (if any)", categorical_columns)
    if selected_category_column:
        selected_category_values = st.sidebar.multiselect(f"Select values for {selected_category_column}", 
                                                          data[selected_category_column].unique())
        data = data[data[selected_category_column].isin(selected_category_values)]
    
    # Filtering by numeric range
    selected_numeric_column = st.sidebar.selectbox("Select Numeric Column to Filter (if any)", numeric_columns)
    if selected_numeric_column:
        min_val, max_val = st.sidebar.slider(f"Select range for {selected_numeric_column}",
                                             float(data[selected_numeric_column].min()), 
                                             float(data[selected_numeric_column].max()), 
                                             (float(data[selected_numeric_column].min()), 
                                              float(data[selected_numeric_column].max())))
        data = data[(data[selected_numeric_column] >= min_val) & (data[selected_numeric_column] <= max_val)]

    # Visualization section
    st.subheader("Visualize Your Data")

    # Chart Generation
    st.subheader("Custom Chart Generator")

    # Select the type of chart
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot"])

    # Select variables for x, y, and hue (if needed)
    x_var = st.selectbox("Select X-axis variable", data.columns)
    y_var = st.selectbox("Select Y-axis variable", data.columns)
    hue_var = st.selectbox("Select hue variable (optional)", [None] + list(data.columns))

    # Convert y_var column to numeric if necessary for numerical aggregation
    data[y_var] = pd.to_numeric(data[y_var], errors='coerce')

    # Generate charts based on selection
    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "Bar Chart":
        sns.barplot(x=x_var, y=y_var, hue=hue_var, data=data, ax=ax)
        ax.set_title(f"{y_var} by {x_var} (Bar Chart)")
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    elif chart_type == "Line Chart":
        sns.lineplot(x=x_var, y=y_var, hue=hue_var, data=data, marker="o", ax=ax)
        ax.set_title(f"{y_var} by {x_var} (Line Chart)")
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    elif chart_type == "Pie Chart":
        # For Pie Chart, aggregate y_var based on categorical variable x_var
        if pd.api.types.is_numeric_dtype(data[y_var]) and pd.api.types.is_categorical_dtype(data[x_var]):
            pie_data = data.groupby(x_var)[y_var].sum()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
            ax.set_title(f"Percentage of {y_var} by {x_var}")
        else:
            st.warning("For Pie charts, select a categorical variable for X and a numeric variable for Y.")

    elif chart_type == "Scatter Plot":
        sns.scatterplot(x=x_var, y=y_var, hue=hue_var, data=data, ax=ax)
        ax.set_title(f"{y_var} vs {x_var} (Scatter Plot)")
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    # Display the chart
    st.pyplot(fig)

    # Sidebar for variable selection for correlation heatmap
    st.sidebar.title("Variable Selection for Correlation Heatmap")
    selected_columns = st.sidebar.multiselect("Select Variables for Correlation Plot", numeric_columns, default=numeric_columns[:3])

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    heatmap_fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data[selected_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(heatmap_fig)
