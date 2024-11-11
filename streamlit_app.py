import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

# Helper function to display visualization guide in the sidebar
def display_visualization_guide():
    st.sidebar.header("Visualization Guide")
    st.sidebar.markdown("""
    **Amounts**: Bar Chart, Dot Plot  
    **Distributions**: Histogram, Density Plot, Boxplot, Violin Plot  
    **Proportions**: Pie Chart, Stacked Bar Chart  
    **Relationships**: Scatter Plot, Line Chart, Bubble Chart  
    **Uncertainty**: Confidence Interval Plot, Error Bars  
    """)

# Function to load and preview data
def load_data():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())
        return df
    else:
        st.write("Please upload a dataset to begin.")
        return None

# Function to create bar and dot plots for categorical vs numerical variables
def visualize_amounts(df):
    st.subheader("Amounts: Bar Chart and Dot Plot")
    
    # Select columns for bar chart
    bar_cat = st.selectbox("Select a categorical variable:", df.select_dtypes(include='object').columns)
    bar_num = st.selectbox("Select a numerical variable:", df.select_dtypes(include='number').columns)
    
    # Bar Chart
    if bar_cat and bar_num:
        st.write("### Bar Chart")
        fig, ax = plt.subplots()
        sns.barplot(x=bar_cat, y=bar_num, data=df, ax=ax)
        ax.set_title(f"Bar Chart of {bar_num} by {bar_cat}")
        st.pyplot(fig)
    
    # Dot Plot
    st.write("### Dot Plot")
    fig, ax = plt.subplots()
    sns.stripplot(x=bar_num, y=bar_cat, data=df, ax=ax, jitter=False)
    ax.set_title(f"Dot Plot of {bar_num} by {bar_cat}")
    st.pyplot(fig)

# Function to create distribution plots
def visualize_distributions(df):
    st.subheader("Distributions: Histogram, Density Plot, Boxplot, Violin Plot")

    # Histogram and Density Plot
    hist_col = st.selectbox("Select a numerical variable for histogram:", df.select_dtypes(include='number').columns)
    hist_color = st.selectbox("Select a categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))
    
    # Histogram
    if hist_col:
        st.write("### Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df, x=hist_col, hue=hist_color, kde=True, ax=ax)
        ax.set_title(f"Histogram of {hist_col}")
        st.pyplot(fig)

    # Boxplot
    st.write("### Boxplot")
    box_cat = st.selectbox("Select categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns))
    fig, ax = plt.subplots()
    sns.boxplot(x=box_cat, y=hist_col, data=df, ax=ax)
    ax.set_title(f"Boxplot of {hist_col}" + (f" grouped by {box_cat}" if box_cat else ""))
    st.pyplot(fig)

    # Violin Plot
    st.write("### Violin Plot")
    fig, ax = plt.subplots()
    sns.violinplot(x=box_cat, y=hist_col, data=df, ax=ax)
    ax.set_title(f"Violin Plot of {hist_col}" + (f" grouped by {box_cat}" if box_cat else ""))
    st.pyplot(fig)

# Function to visualize relationships between two variables (Scatter and Line Charts)
def visualize_relationships(df):
    st.subheader("X-Y Relationships: Scatter Plot, Line Chart")
    
    # Scatter Plot
    scatter_x = st.selectbox("Select X-axis variable:", df.select_dtypes(include='number').columns)
    scatter_y = st.selectbox("Select Y-axis variable:", df.select_dtypes(include='number').columns)
    if scatter_x and scatter_y:
        st.write("### Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(x=scatter_x, y=scatter_y, data=df, ax=ax)
        ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
        st.pyplot(fig)
    
    # Line Chart
    st.write("### Line Chart")
    line_x = st.selectbox("Select X-axis for line chart:", df.columns)
    line_y = st.selectbox("Select Y-axis for line chart:", df.select_dtypes(include='number').columns)
    if line_x and line_y:
        fig, ax = plt.subplots()
        sns.lineplot(x=line_x, y=line_y, data=df, ax=ax)
        ax.set_title(f"Line Chart of {line_y} over {line_x}")
        st.pyplot(fig)

# Function to visualize three-variable relationships (Bubble Chart, 3D Scatter)
def visualize_three_variable_relationships(df):
    st.subheader("Three-Variable Relationships: Bubble Chart and 3D Scatter")

    # Bubble Chart
    bubble_x = st.selectbox("Select X-axis variable for bubble chart:", df.select_dtypes(include='number').columns)
    bubble_y = st.selectbox("Select Y-axis variable for bubble chart:", df.select_dtypes(include='number').columns)
    bubble_size = st.selectbox("Select variable for bubble size:", df.select_dtypes(include='number').columns)
    bubble_color = st.selectbox("Select categorical variable for bubble color (optional):", [None] + list(df.select_dtypes(include='object').columns))

    if bubble_x and bubble_y and bubble_size:
        st.write("### Bubble Chart")
        fig, ax = plt.subplots()
        sns.scatterplot(x=bubble_x, y=bubble_y, size=bubble_size, hue=bubble_color, data=df, ax=ax)
        ax.set_title(f"Bubble Chart of {bubble_x} vs {bubble_y} with bubble size based on {bubble_size}")
        st.pyplot(fig)

# Main function to run the app
def main():
    st.title("Enhanced Exploratory Data Analysis App")
    st.write("Upload a dataset to explore its variables with various visualizations.")
    display_visualization_guide()
    df = load_data()
    
    if df is not None:
        st.header("One Variable Analysis")
        visualize_amounts(df)
        visualize_distributions(df)
        
        st.header("Two Variable Analysis")
        visualize_relationships(df)
        
        st.header("Three Variable Analysis")
        visualize_three_variable_relationships(df)

if __name__ == "__main__":
    main()
