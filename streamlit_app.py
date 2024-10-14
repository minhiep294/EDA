import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset uploaded by the user
st.title("📊 Generalized EDA Dashboard")

uploaded_data_file = st.file_uploader("Upload a dataset (.csv)", type=["csv"])

if uploaded_data_file:
    # Read the dataset
    data = pd.read_csv(uploaded_data_file)

    # Sidebar for filtering
    st.sidebar.title("Filter Options")

    # Dynamically get the column types (numeric, categorical)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Multi-select for categorical filtering
    filters = {}
    for col in categorical_columns:
        unique_values = data[col].unique().tolist()
        selected_values = st.sidebar.multiselect(f"Filter by {col}", unique_values, default=unique_values)
        filters[col] = selected_values

    # Filter the data based on the user input
    for col, selected_values in filters.items():
        data = data[data[col].isin(selected_values)]

    # Numeric sliders for filtering numeric columns
    for col in numeric_columns:
        min_val, max_val = data[col].min(), data[col].max()
        selected_range = st.sidebar.slider(f"Range for {col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
        data = data[(data[col] >= selected_range[0]) & (data[col] <= selected_range[1])]

    # Metrics Overview for the entire dataset
    st.subheader("Dataset Overview")
    st.write(f"Number of rows: {len(data)}")
    st.write(f"Number of columns: {data.shape[1]}")
    st.write("Data types:")
    st.write(data.dtypes)

    # Show summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Visualization section
    st.subheader("Visualize Your Data")

    # Feature selection for visualization
    if numeric_columns:
        st.subheader('Select Features for Visualization')

        x_axis = st.selectbox('Select X-axis feature (numeric)', options=numeric_columns, index=0)
        y_axis = st.selectbox('Select Y-axis feature (numeric)', options=numeric_columns, index=1)

        # Visualization type selection
        plot_type = st.selectbox("Select Visualization Type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram"])

        # Generate plots based on user selection
        if plot_type == "Scatter Plot":
            st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=categorical_columns[0] if categorical_columns else None, ax=ax)
            plt.title(f"Scatter Plot of {x_axis} vs {y_axis}")
            st.pyplot(fig)

        elif plot_type == "Line Plot":
            st.subheader(f"Line Plot: {x_axis} vs {y_axis}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=data, x=x_axis, y=y_axis, hue=categorical_columns[0] if categorical_columns else None, ax=ax)
            plt.title(f"Line Plot of {x_axis} vs {y_axis}")
            st.pyplot(fig)

        elif plot_type == "Bar Plot":
            st.subheader(f"Bar Plot: {x_axis} vs {y_axis}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=data, x=x_axis, y=y_axis, hue=categorical_columns[0] if categorical_columns else None, ax=ax)
            plt.title(f"Bar Plot of {x_axis} vs {y_axis}")
            st.pyplot(fig)

        elif plot_type == "Histogram":
            st.subheader(f"Histogram of {x_axis}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[x_axis], bins=30, kde=True, ax=ax)
            plt.title(f"Histogram of {x_axis}")
            st.pyplot(fig)

    else:
        st.warning("No numeric columns available for plotting.")

    # Display filtered data table
    st.subheader("Filtered Data Table")
    st.dataframe(data)
