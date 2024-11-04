import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to display visualization guide based on data type
def display_visualization_guide():
    st.sidebar.header("Visualization Guide")
    st.sidebar.markdown("""
    - **Amounts**: Bar charts, dot plots, stacked bar charts.
    - **Distributions**: Histograms, density plots, boxplots, violin plots, cumulative density plots.
    - **Proportions**: Pie charts, stacked bars, mosaic plots, treemaps.
    - **X–Y Relationships**: Scatterplots, line charts, correlation heatmaps.
    - **Geospatial Data**: Choropleth maps, cartograms.
    - **Uncertainty**: Error bars, confidence bands, quantile dot plots.
    """)

# App title and file upload
st.title("Enhanced Data Exploration with Streamlit")
st.write("Upload a dataset to explore its variables with various charts and aesthetics.")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    display_visualization_guide()
else:
    st.write("Please upload a dataset to begin.")

# Section 1: Visualizing Amounts
st.header("Visualizing Amounts")

# Bar Chart
st.subheader("Bar Chart")
st.write("Useful for visualizing amounts across categories.")
col_bar_cat = st.selectbox("Select a categorical variable for the bar chart:", df.select_dtypes(include='object').columns)
col_bar_num = st.selectbox("Select a numerical variable for the bar chart:", df.select_dtypes(include='number').columns)
bar_color = st.selectbox("Select a second categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))

if col_bar_cat and col_bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=col_bar_cat, y=col_bar_num, hue=bar_color, data=df, ax=ax)
    ax.set_title(f"Bar Chart of {col_bar_num} by {col_bar_cat}")
    st.pyplot(fig)

# Dot Plot (Alternative to Bar Chart)
st.subheader("Dot Plot")
st.write("Alternative to bar chart for visualizing amounts.")
if col_bar_cat and col_bar_num:
    fig, ax = plt.subplots()
    sns.stripplot(x=col_bar_num, y=col_bar_cat, data=df, ax=ax, jitter=False)
    ax.set_title(f"Dot Plot of {col_bar_num} by {col_bar_cat}")
    st.pyplot(fig)

# Section 2: Visualizing Distributions
st.header("Visualizing Distributions")

# Histogram with Density Plot
st.subheader("Histogram with Density")
st.write("Select a numerical variable to view its distribution with density overlay.")
col_hist = st.selectbox("Select a numerical variable for the histogram:", df.select_dtypes(include='number').columns)
hist_color = st.selectbox("Select a categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))
bins = st.slider("Number of bins:", 5, 50, 10)

if col_hist:
    fig, ax = plt.subplots()
    sns.histplot(df, x=col_hist, hue=hist_color, bins=bins, kde=True, ax=ax)
    ax.set_title(f"Histogram of {col_hist}")
    st.pyplot(fig)

# Cumulative Density Plot
st.subheader("Cumulative Density Plot")
st.write("Displays cumulative density for a numerical variable.")
if col_hist:
    fig, ax = plt.subplots()
    sns.ecdfplot(df, x=col_hist, hue=hist_color, ax=ax)
    ax.set_title(f"Cumulative Density Plot of {col_hist}")
    st.pyplot(fig)

# Section 3: Visualizing Proportions
st.header("Visualizing Proportions")

# Pie Chart
st.subheader("Pie Chart")
st.write("Displays proportions of categories within a categorical variable.")
col_pie = st.selectbox("Select a categorical variable for pie chart:", df.select_dtypes(include='object').columns)

if col_pie:
    fig, ax = plt.subplots()
    df[col_pie].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_title(f"Pie Chart of {col_pie}")
    st.pyplot(fig)

# Stacked Bar Chart for Proportions
st.subheader("Stacked Bar Chart for Proportions")
st.write("Displays proportions across two categorical variables.")
col_stack_cat1 = st.selectbox("Select the first categorical variable:", df.select_dtypes(include='object').columns, key="stack_cat1")
col_stack_cat2 = st.selectbox("Select the second categorical variable:", df.select_dtypes(include='object').columns, key="stack_cat2")

if col_stack_cat1 and col_stack_cat2 and col_stack_cat1 != col_stack_cat2:
    stacked_data = df.groupby([col_stack_cat1, col_stack_cat2]).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    stacked_data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"Stacked Bar Chart of {col_stack_cat1} by {col_stack_cat2}")
    ax.set_xlabel(col_stack_cat1)
    st.pyplot(fig)

# Section 4: X-Y Relationships
st.header("X-Y Relationships")

# Scatter Plot with Color and Size Aesthetics
st.subheader("Scatter Plot")
st.write("Select two numerical variables to visualize relationships.")
scatter_x = st.selectbox("Select X-axis variable:", df.select_dtypes(include='number').columns)
scatter_y = st.selectbox("Select Y-axis variable:", df.select_dtypes(include='number').columns)
scatter_color = st.selectbox("Select a categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))
scatter_size = st.selectbox("Select numerical variable for size (optional):", [None] + list(df.select_dtypes(include='number').columns))

if scatter_x and scatter_y:
    fig, ax = plt.subplots()
    sns.scatterplot(x=scatter_x, y=scatter_y, hue=scatter_color, size=scatter_size, data=df, ax=ax)
    ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
    st.pyplot(fig)

# Line Chart for Time Series
st.subheader("Line Chart")
st.write("Ideal for visualizing trends over time or continuous sequences.")
time_col = st.selectbox("Select a time column for x-axis (optional):", [None] + list(df.columns))
line_y = st.selectbox("Select a numerical variable for y-axis:", df.select_dtypes(include='number').columns)

if time_col and line_y:
    fig, ax = plt.subplots()
    sns.lineplot(x=df[time_col], y=df[line_y], ax=ax)
    ax.set_title(f"Line Chart of {line_y} over {time_col}")
    st.pyplot(fig)

# Section 5: Visualizing Uncertainty
st.header("Visualizing Uncertainty")

# Confidence Interval Plot
st.subheader("Confidence Interval Plot")
st.write("Visualizes mean and confidence interval for a numerical variable.")
conf_num = st.selectbox("Select numerical variable for CI plot:", df.select_dtypes(include='number').columns)
conf_cat = st.selectbox("Select categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns))

if conf_num:
    fig, ax = plt.subplots()
    sns.pointplot(x=conf_cat, y=conf_num, data=df, ci='sd', ax=ax)
    ax.set_title(f"Confidence Interval Plot of {conf_num}" + (f" grouped by {conf_cat}" if conf_cat else ""))
    st.pyplot(fig)
