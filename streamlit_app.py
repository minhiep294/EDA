import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to generate basic statistical analysis description
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
st.title("Basic EDA Tool")

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

    # Basic Summary Statistics
    if st.checkbox("Show Summary Statistics"):
        st.write("Summary Statistics")
        st.write(data.describe())

    # Variable Selection
    st.write("Select variables to visualize:")
    selected_vars = st.multiselect("Choose up to two variables for plotting:", data.columns, max_selections=2)
    
    # Plot Type Selection based on the number of selected variables
    if len(selected_vars) == 1:
        st.write("### Single Variable Visualization")
        plot_type = st.selectbox("Select plot type:", ["Line Chart", "Histogram", "Bar Chart (Categorical)"])

        plt.figure(figsize=(10, 6))
        feature = selected_vars[0]
        
        # Numeric Visualizations
        if data[feature].dtype in [np.number, 'float64', 'int64']:
            if plot_type == "Line Chart":
                plt.plot(data[feature])
                plt.title(f'Line Chart of {feature}')
                plt.xlabel('Index')
                plt.ylabel(feature)

            elif plot_type == "Histogram":
                sns.histplot(data[feature], kde=True)
                plt.title(f'Histogram of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')

        # Categorical Visualization
        elif data[feature].dtype == 'object' and plot_type == "Bar Chart (Categorical)":
            data[feature].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')

        st.pyplot(plt)

    elif len(selected_vars) == 2:
        st.write("### Two Variable Visualization")
        plot_type = st.selectbox("Select plot type:", ["Scatter Plot", "Line Chart"])

        x_axis, y_axis = selected_vars

        plt.figure(figsize=(10, 6))

        if plot_type == "Scatter Plot":
            sns.scatterplot(data=data, x=x_axis, y=y_axis)
            plt.title(f'Scatter Plot of {y_axis} vs {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

        elif plot_type == "Line Chart":
            plt.plot(data[x_axis], data[y_axis])
            plt.title(f'Line Chart of {y_axis} vs {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

        st.pyplot(plt)

    # AI Analysis
    st.subheader("AI Analysis")
    selected_feature = st.selectbox("Select feature for AI analysis:", data.columns)
    analysis_description = generate_analysis(selected_feature, data)
    st.write(analysis_description)
