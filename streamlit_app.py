import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to display visualization guide based on data type
def display_visualization_guide():
    st.sidebar.header("Visualization Guide")
    st.sidebar.markdown("""
    - **Amounts**: Bar charts, dot plots, stacked bar charts.
    - **Distributions**: Histograms, density plots, cumulative density plots, boxplots.
    - **Proportions**: Pie charts, side-by-side bars, stacked bars, mosaic plots.
    - **X–Y Relationships**: Scatterplots, line charts, correlation heatmaps.
    - **Geospatial Data**: Choropleth maps (for spatial data with latitude and longitude).
    - **Uncertainty**: Error bars, confidence intervals, confidence bands.
    """)

# App title and file upload
st.title("Exploratory Data Analysis App")
st.write("Upload a dataset to explore its variables with various charts and aesthetics.")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    display_visualization_guide()
else:
    st.write("Please upload a dataset to begin.")

# Section 1: Amounts
st.header("Visualizing Amounts")
st.write("Use bar charts or dot plots to display numerical values for categories.")

# Bar Chart
st.subheader("Bar Chart")
st.write("Displays amounts for each category.")
bar_cat = st.selectbox("Select categorical variable for bar chart:", df.select_dtypes(include='object').columns)
bar_num = st.selectbox("Select numerical variable for bar chart:", df.select_dtypes(include='number').columns)

if bar_cat and bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=bar_cat, y=bar_num, data=df, ax=ax)
    ax.set_title(f"Bar Chart of {bar_num} by {bar_cat}")
    st.pyplot(fig)

# Dot Plot (Alternative to Bar Chart)
st.subheader("Dot Plot")
st.write("An alternative to bar chart for visualizing amounts.")
if bar_cat and bar_num:
    fig, ax = plt.subplots()
    sns.stripplot(x=bar_num, y=bar_cat, data=df, ax=ax, jitter=False)
    ax.set_title(f"Dot Plot of {bar_num} by {bar_cat}")
    st.pyplot(fig)

# Section 2: Distributions
st.header("Visualizing Distributions")
st.write("Explore the distribution of numerical data using histograms, density plots, and boxplots.")

# Histogram
st.subheader("Histogram with Density Overlay")
col_hist = st.selectbox("Select numerical variable for histogram:", df.select_dtypes(include='number').columns)
bins = st.slider("Number of bins:", 5, 50, 10)

if col_hist:
    fig, ax = plt.subplots()
    sns.histplot(df[col_hist], bins=bins, kde=True, ax=ax)
    ax.set_title(f"Histogram of {col_hist}")
    st.pyplot(fig)

# Boxplot
st.subheader("Boxplot")
col_box = st.selectbox("Select numerical variable for boxplot:", df.select_dtypes(include='number').columns)
col_box_cat = st.selectbox("Select categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns))

if col_box:
    fig, ax = plt.subplots()
    sns.boxplot(x=col_box_cat, y=col_box, data=df, ax=ax)
    ax.set_title(f"Boxplot of {col_box}" + (f" grouped by {col_box_cat}" if col_box_cat else ""))
    st.pyplot(fig)

# Section 3: Proportions
st.header("Visualizing Proportions")
st.write("Display proportions of categories with pie charts or stacked bar charts.")

# Pie Chart
st.subheader("Pie Chart")
col_pie = st.selectbox("Select categorical variable for pie chart:", df.select_dtypes(include='object').columns)

if col_pie:
    fig, ax = plt.subplots()
    df[col_pie].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_title(f"Pie Chart of {col_pie}")
    st.pyplot(fig)

# Stacked Bar Chart
st.subheader("Stacked Bar Chart")
col_stack_cat1 = st.selectbox("Select first categorical variable for stacked bar chart:", df.select_dtypes(include='object').columns, key="stack_cat1")
col_stack_cat2 = st.selectbox("Select second categorical variable:", df.select_dtypes(include='object').columns, key="stack_cat2")

if col_stack_cat1 and col_stack_cat2:
    stacked_data = df.groupby([col_stack_cat1, col_stack_cat2]).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    stacked_data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"Stacked Bar Chart of {col_stack_cat1} by {col_stack_cat2}")
    st.pyplot(fig)

# Section 4: X-Y Relationships
st.header("Visualizing X-Y Relationships")
st.write("Investigate relationships between two variables with scatter plots, line charts, and correlation heatmaps.")

# Scatter Plot
st.subheader("Scatter Plot")
scatter_x = st.selectbox("Select X-axis variable:", df.select_dtypes(include='number').columns)
scatter_y = st.selectbox("Select Y-axis variable:", df.select_dtypes(include='number').columns)

if scatter_x and scatter_y:
    fig, ax = plt.subplots()
    sns.scatterplot(x=scatter_x, y=scatter_y, data=df, ax=ax)
    ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
    st.pyplot(fig)

# Line Chart
st.subheader("Line Chart")
line_y = st.selectbox("Select numerical variable for y-axis:", df.select_dtypes(include='number').columns)
time_col = st.selectbox("Select time or index column for x-axis (optional):", [None] + list(df.columns))

if line_y and time_col:
    fig, ax = plt.subplots()
    sns.lineplot(x=df[time_col], y=df[line_y], ax=ax)
    ax.set_title(f"Line Chart of {line_y} over {time_col}")
    st.pyplot(fig)

# Section 5: Uncertainty
st.header("Visualizing Uncertainty")
st.write("Use confidence intervals or error bars to represent uncertainty in data.")

# Confidence Interval Plot
st.subheader("Confidence Interval Plot")
conf_num = st.selectbox("Select numerical variable for CI plot:", df.select_dtypes(include='number').columns)
conf_cat = st.selectbox("Select categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns))

if conf_num:
    fig, ax = plt.subplots()
    sns.pointplot(x=conf_cat, y=conf_num, data=df, ci='sd', ax=ax)
    ax.set_title(f"Confidence Interval Plot of {conf_num}" + (f" grouped by {conf_cat}" if conf_cat else ""))
    st.pyplot(fig)

# Section 6: Three Variable Analysis
st.header("Three Variable Analysis")
st.write("Explore relationships between multiple variables with pair plots.")

# Pair Plot
pair_cols = st.multiselect("Select numerical variables for pair plot:", df.select_dtypes(include='number').columns)
pair_hue = st.selectbox("Select categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))

if len(pair_cols) >= 2:
    fig = sns.pairplot(df, vars=pair_cols, hue=pair_hue)
    st.pyplot(fig)
