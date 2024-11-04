import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates

# Function to generate analysis description
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

# App title
st.title("EDA Tool")

# Upload Data section
st.subheader("Upload Data File")
uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read data from the uploaded file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file, encoding='windows-1252')
    else:
        data = pd.read_excel(uploaded_file, engine='openpyxl')
    
    st.write("Data has been uploaded:")
    st.dataframe(data)

    # Filter Section for Year and Bank Code
    st.subheader("Filter Data by Year and Bank Code")

    # Filter by Year if the column exists
    if 'Year' in data.columns:
        unique_years = sorted(data['Year'].dropna().unique())
        selected_years = st.multiselect("Select Year(s):", unique_years, default=unique_years)
        filtered_data = data[data['Year'].isin(selected_years)]
    else:
        filtered_data = data.copy()

    # Filter by Bank Code if the column exists
    if 'Bank Code' in filtered_data.columns:
        unique_bank_codes = filtered_data['Bank Code'].dropna().unique()
        selected_bank_codes = st.multiselect("Select Bank Code(s):", unique_bank_codes, default=unique_bank_codes)
        filtered_data = filtered_data[filtered_data['Bank Code'].isin(selected_bank_codes)]

    st.write("Filtered Data Preview:")
    st.dataframe(filtered_data)

    # Analysis Options
    st.subheader("Data Analysis and Visualization")

    # Display Summary Statistics
    if st.checkbox("Show Summary Statistics"):
        st.write("Summary Statistics")
        st.write(filtered_data.describe())

    # Variable Selection
    st.write("Select up to three variables to plot:")
    selected_vars = st.multiselect("Variables:", filtered_data.columns, max_selections=3)
    
    # Plot Type Selection based on the number of selected variables
    if len(selected_vars) == 1:
        st.write("### Single Variable Visualization")
        plot_type = st.selectbox("Select plot type:", [
            "Line Chart", "Histogram", "Box Plot", 
            "Bar Chart (Categorical)", "Pie Chart (Categorical)"
        ])

        plt.figure(figsize=(10, 6))
        feature = selected_vars[0]
        
        # Numeric Visualizations
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
                plt.title(f'Boxplot of {feature}')

        # Categorical Visualizations
        elif filtered_data[feature].dtype == 'object':
            if plot_type == "Bar Chart (Categorical)":
                filtered_data[feature].value_counts().plot(kind='bar')
                plt.title(f'Bar Chart of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Count')

            elif plot_type == "Pie Chart (Categorical)":
                filtered_data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {feature}')

        st.pyplot(plt)

    elif len(selected_vars) == 2:
        st.write("### Two Variable Visualization")
        plot_type = st.selectbox("Select plot type:", [
            "Scatter Plot", "Box Plot", "Line Graph", "Grouped Bar Chart"
        ])

        x_axis, y_axis = selected_vars

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
