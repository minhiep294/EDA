import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("Data Visualization Tool")
st.markdown("### Following Best Practices from *Fundamentals of Data Visualization*")

# Sidebar for instructions and guidelines
st.sidebar.header("Guidelines")
st.sidebar.markdown("""
- Choose suitable chart types based on your data and analytical needs.
- Customize chart options, such as labels and colors, for clarity and readability.
- Avoid cluttering charts with unnecessary details; aim for simplicity and interpretability.
""")

# Upload Dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
else:
    st.write("Please upload a dataset to begin.")

# Display requirements for chart types
def display_chart_requirements():
    st.sidebar.markdown("""
    ### Chart Requirements:
    - **Histogram**: One numerical variable
    - **Boxplot**: One numerical variable (optional: categorical for grouping)
    - **Scatter Plot**: Two numerical variables
    - **Line Plot**: One or two numerical variables with optional time variable
    - **Bar Plot**: One categorical and one numerical variable
    - **Density Plot**: Two numerical variables for joint distribution
    """)

display_chart_requirements()

# Single Variable Analysis - Histogram and Boxplot
st.header("Single Variable Analysis")

# Histogram
st.subheader("Histogram")
st.write("A histogram displays the distribution of a single numerical variable.")
col_hist = st.selectbox("Select a numerical variable for histogram:", df.select_dtypes(include='number').columns)
bins = st.slider("Number of bins:", 5, 50, 10)
color_hist = st.color_picker("Pick a color for the histogram", "#1f77b4")

if col_hist:
    fig, ax = plt.subplots()
    sns.histplot(df[col_hist], bins=bins, kde=True, color=color_hist, ax=ax)
    ax.set_title(f"Histogram of {col_hist}")
    ax.set_xlabel(col_hist)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Boxplot
st.subheader("Boxplot")
st.write("Boxplots display the spread and outliers of a numerical variable.")
col_box = st.selectbox("Select a numerical variable for boxplot:", df.select_dtypes(include='number').columns)
col_box_cat = st.selectbox("Select a categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns))

if col_box:
    fig, ax = plt.subplots()
    if col_box_cat:
        sns.boxplot(x=col_box_cat, y=col_box, data=df, ax=ax)
        ax.set_title(f"Boxplot of {col_box} grouped by {col_box_cat}")
    else:
        sns.boxplot(y=df[col_box], ax=ax)
        ax.set_title(f"Boxplot of {col_box}")
    ax.set_xlabel(col_box_cat if col_box_cat else "")
    ax.set_ylabel(col_box)
    st.pyplot(fig)

# Two Variable Analysis - Scatter Plot, Line Plot, and Bar Plot
st.header("Two Variable Analysis")

# Scatter Plot
st.subheader("Scatter Plot")
st.write("Use scatter plots to show relationships between two numerical variables.")
col_scatter_x = st.selectbox("Select X-axis variable for scatter plot:", df.select_dtypes(include='number').columns)
col_scatter_y = st.selectbox("Select Y-axis variable for scatter plot:", df.select_dtypes(include='number').columns)
color_scatter = st.color_picker("Pick a color for scatter points", "#1f77b4")

if col_scatter_x and col_scatter_y and col_scatter_x != col_scatter_y:
    fig, ax = plt.subplots()
    sns.scatterplot(x=col_scatter_x, y=col_scatter_y, data=df, color=color_scatter, ax=ax)
    ax.set_title(f"Scatter Plot of {col_scatter_x} vs {col_scatter_y}")
    ax.set_xlabel(col_scatter_x)
    ax.set_ylabel(col_scatter_y)
    st.pyplot(fig)

# Line Plot
st.subheader("Line Plot")
st.write("Line plots are useful for time series or trends over a sequential index.")
col_line_y = st.selectbox("Select a numerical variable for line plot:", df.select_dtypes(include='number').columns, key="line_y")
time_col = st.selectbox("Select a time or index column (optional):", [None] + list(df.columns), key="time_col_line")
color_line = st.color_picker("Pick a color for the line", "#1f77b4")

if col_line_y:
    fig, ax = plt.subplots()
    if time_col:
        sns.lineplot(x=df[time_col], y=df[col_line_y], color=color_line, ax=ax)
        ax.set_title(f"Line Plot of {col_line_y} over {time_col}")
        ax.set_xlabel(time_col)
    else:
        sns.lineplot(data=df[col_line_y], color=color_line, ax=ax)
        ax.set_title(f"Line Plot of {col_line_y} over Index")
        ax.set_xlabel("Index")
    ax.set_ylabel(col_line_y)
    st.pyplot(fig)

# Bar Plot
st.subheader("Bar Plot")
st.write("Bar plots are useful for summarizing numerical data by a categorical variable.")
col_bar_cat = st.selectbox("Select a categorical variable for bar plot:", df.select_dtypes(include='object').columns)
col_bar_num = st.selectbox("Select a numerical variable for bar plot:", df.select_dtypes(include='number').columns)
color_bar = st.color_picker("Pick a color for the bars", "#1f77b4")

if col_bar_cat and col_bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=col_bar_cat, y=col_bar_num, data=df, color=color_bar, ax=ax)
    ax.set_title(f"Bar Plot of {col_bar_num} by {col_bar_cat}")
    ax.set_xlabel(col_bar_cat)
    ax.set_ylabel(col_bar_num)
    st.pyplot(fig)

# Density Plot
st.subheader("Density Plot")
st.write("Use density plots to show the joint distribution of two numerical variables.")
col_density_x = st.selectbox("Select X-axis variable for density plot:", df.select_dtypes(include='number').columns, key="density_x")
col_density_y = st.selectbox("Select Y-axis variable for density plot:", df.select_dtypes(include='number').columns, key="density_y")

if col_density_x and col_density_y and col_density_x != col_density_y:
    fig, ax = plt.subplots()
    sns.kdeplot(x=df[col_density_x], y=df[col_density_y], cmap="Blues", ax=ax, fill=True)
    ax.set_title(f"Density Plot of {col_density_x} and {col_density_y}")
    ax.set_xlabel(col_density_x)
    ax.set_ylabel(col_density_y)
    st.pyplot(fig)
