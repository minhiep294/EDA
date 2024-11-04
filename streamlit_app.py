import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates

# Function to generate a basic analysis description
def generate_analysis(feature, data):
    data[feature] = pd.to_numeric(data[feature], errors='coerce')
    mean_value = data[feature].mean()
    std_value = data[feature].std()
    median_value = data[feature].median()
    trend = "increasing" if data[feature].iloc[-1] > data[feature].iloc[0] else "decreasing"
    
    description = [
        f"The mean of {feature} is {mean_value:.2f}, with a standard deviation of {std_value:.2f}.",
        f"The median value of {feature} is {median_value:.2f}.",
        f"The trend is {trend} over the selected period.",
        f"This indicates that {feature} has shown a {trend} trend recently."
    ]

    return " ".join(description)

# App title and description
st.title("Comprehensive EDA Tool")
st.markdown("### A structured, flexible approach to exploring any dataset.")

# Step 1: Upload Data
st.subheader("Step 1: Upload Your Data File")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read data from uploaded file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file, encoding='windows-1252')
    else:
        data = pd.read_excel(uploaded_file, engine='openpyxl')
    
    st.write("Data successfully uploaded!")
    st.dataframe(data.head())

    # Step 2: Filter Data by Key Columns
    st.subheader("Step 2: Filter Data by Key Columns")

    # Filter by "Year" if it exists in the dataset
    filtered_data = data.copy()
    if 'Year' in data.columns:
        unique_years = sorted(data['Year'].dropna().unique())
        selected_years = st.multiselect("Select Year(s):", unique_years, default=unique_years)
        filtered_data = filtered_data[filtered_data['Year'].isin(selected_years)]
    
    # Filter by "Bank Code" if it exists in the dataset
    if 'Bank Code' in filtered_data.columns:
        unique_bank_codes = filtered_data['Bank Code'].dropna().unique()
        selected_bank_codes = st.multiselect("Select Bank Code(s):", unique_bank_codes, default=unique_bank_codes)
        filtered_data = filtered_data[filtered_data['Bank Code'].isin(selected_bank_codes)]
    
    st.write("Filtered Data Preview:")
    st.dataframe(filtered_data.head())

    # Step 3: Display Summary Statistics
    st.subheader("Step 3: View Summary Statistics")
    if st.checkbox("Show Summary Statistics"):
        st.write("Summary Statistics:")
        st.write(filtered_data.describe())

    # Step 4: Variable Analysis
    st.subheader("Step 4: Analyze Individual Variables")
    selected_feature = st.selectbox("Select feature for AI-driven analysis:", filtered_data.columns)
    analysis_description = generate_analysis(selected_feature, filtered_data)
    st.write(analysis_description)

    # Step 5: Data Visualization
    st.subheader("Step 5: Data Visualization")

    # Select variables for visualization
    selected_vars = st.multiselect("Choose up to three variables to visualize:", filtered_data.columns, max_selections=3)
    
    if len(selected_vars) == 1:
        st.write("### Single Variable Visualization")
        plot_type = st.selectbox("Select plot type:", ["Line Chart", "Histogram", "Box Plot", "Bar Chart", "Pie Chart"])

        feature = selected_vars[0]
        plt.figure(figsize=(10, 6))
        
        if filtered_data[feature].dtype in [np.number, 'float64', 'int64']:
            if plot_type == "Line Chart":
                plt.plot(filtered_data[feature])
                plt.title(f'Line Chart of {feature}')
                plt.xlabel('Index')
                plt.ylabel(feature)

            elif plot_type == "Histogram":
                sns.histplot(filtered_data[feature], kde=True)
                plt.title(f'Histogram of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')

            elif plot_type == "Box Plot":
                sns.boxplot(y=filtered_data[feature])
                plt.title(f'Box Plot of {feature}')

        elif filtered_data[feature].dtype == 'object':
            if plot_type == "Bar Chart":
                filtered_data[feature].value_counts().plot(kind='bar')
                plt.title(f'Bar Chart of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Count')

            elif plot_type == "Pie Chart":
                filtered_data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {feature}')
        
        st.pyplot(plt)

    elif len(selected_vars) == 2:
        st.write("### Two Variable Visualization")
        
        # Select x and y axes
        x_axis = st.selectbox("Select X-axis variable:", selected_vars)
        y_axis = st.selectbox("Select Y-axis variable:", [var for var in selected_vars if var != x_axis])

        plot_type = st.selectbox("Select plot type:", ["Scatter Plot", "Box Plot", "Line Graph", "Grouped Bar Chart"])

        plt.figure(figsize=(10, 6))

        if plot_type == "Scatter Plot":
            sns.scatterplot(data=filtered_data, x=x_axis, y=y_axis)
            plt.title(f'Scatter Plot of {y_axis} vs {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

        elif plot_type == "Box Plot":
            sns.boxplot(data=filtered_data, x=x_axis, y=y_axis)
            plt.title(f'Box Plot of {y_axis} by {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

        elif plot_type == "Line Graph":
            plt.plot(filtered_data[x_axis], filtered_data[y_axis])
            plt.title(f'Line Graph of {y_axis} vs {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

        elif plot_type == "Grouped Bar Chart":
            filtered_data.groupby(x_axis)[y_axis].mean().plot(kind='bar')
            plt.title(f'Grouped Bar Chart of {y_axis} by {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(f'Mean {y_axis}')

        st.pyplot(plt)

    elif len(selected_vars) == 3:
        st.write("### Three Variable Visualization")
        plot_type = st.selectbox("Select plot type:", ["3D Scatter Plot", "Parallel Coordinates Plot"])

        if plot_type == "3D Scatter Plot":
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(filtered_data[selected_vars[0]], filtered_data[selected_vars[1]], filtered_data[selected_vars[2]])
            ax.set_xlabel(selected_vars[0])
            ax.set_ylabel(selected_vars[1])
            ax.set_zlabel(selected_vars[2])
            plt.title('3D Scatter Plot of Selected Variables')
            st.pyplot(plt)
        
        elif plot_type == "Parallel Coordinates Plot":
            plt.figure(figsize=(10, 6))
            parallel_coordinates(filtered_data[selected_vars], class_column=selected_vars[0])
            plt.title('Parallel Coordinates Plot')
            st.pyplot(plt)
