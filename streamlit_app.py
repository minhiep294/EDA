import streamlit as st
import pandas as pd
import numpy as np

def dynamic_filter(df):
    """
    Provides a dynamic filtering interface for a pandas DataFrame.
    Users can add filters dynamically for selected columns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to filter.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    st.sidebar.header("Dynamic Filter Builder")
    filter_columns = st.sidebar.multiselect("Choose columns to filter:", df.columns)

    filter_conditions = {}  # Dictionary to store filter conditions
    
    for col in filter_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Slider for numerical columns
            min_val, max_val = df[col].min(), df[col].max()
            range_filter = st.sidebar.slider(
                f"Filter {col} (Numeric Range):",
                float(min_val), float(max_val),
                (float(min_val), float(max_val))
            )
            filter_conditions[col] = range_filter
        
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Multiselect for categorical columns
            unique_values = df[col].dropna().unique()
            selected_values = st.sidebar.multiselect(
                f"Filter {col} (Select Categories):",
                unique_values, default=unique_values
            )
            if selected_values:
                filter_conditions[col] = selected_values
        
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Date range picker for datetime columns
            min_date, max_date = df[col].min(), df[col].max()
            date_range = st.sidebar.date_input(
                f"Filter {col} (Date Range):",
                (min_date, max_date), min_value=min_date, max_value=max_date
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                filter_conditions[col] = date_range

    # Apply filters
    filtered_df = df.copy()
    for col, condition in filter_conditions.items():
        if isinstance(condition, tuple):  # For numerical ranges
            filtered_df = filtered_df[(filtered_df[col] >= condition[0]) & (filtered_df[col] <= condition[1])]
        elif isinstance(condition, list):  # For categorical selections
            filtered_df = filtered_df[filtered_df[col].isin(condition)]
        elif isinstance(condition, pd.Timestamp):  # For datetime ranges
            filtered_df = filtered_df[(filtered_df[col] >= pd.Timestamp(condition[0])) &
                                      (filtered_df[col] <= pd.Timestamp(condition[1]))]
    
    return filtered_df

# Example Usage
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Age": np.random.randint(20, 50, 5),
        "Salary": np.random.randint(40000, 100000, 5),
        "Joining Date": pd.date_range("2020-01-01", periods=5, freq="365D"),
        "Department": ["HR", "Finance", "IT", "Finance", "HR"]
    }
    df = pd.DataFrame(data)

    st.title("Dynamic Filter Example")
    st.write("### Original Data:")
    st.dataframe(df)

    # Apply dynamic filter
    filtered_df = dynamic_filter(df)

    st.write("### Filtered Data:")
    st.dataframe(filtered_df)
