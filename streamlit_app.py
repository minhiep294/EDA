import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper functions for charts with brief guidelines
def display_guidelines():
    st.sidebar.header("Chart Requirements & Guidelines")
    st.sidebar.markdown("""
        - **Histogram**: Requires a single numerical variable.
        - **Boxplot**: Requires a single numerical variable, and optionally, a categorical variable for grouping.
        - **Scatter Plot**: Requires two numerical variables.
        - **Correlation Heatmap**: Requires at least two numerical variables.
        - **Bar Plot**: Requires one categorical and one numerical variable.
        - **Line Plot**: Requires one or two numerical variables with an optional time/index variable.
        - **Density Plot**: Requires two numerical variables to show joint distribution.
        - **Hexbin Plot**: Requires two numerical variables, useful for large datasets.
    """)

# Load data and display sample
st.title("Flexible Data Exploration with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    display_guidelines()
else:
    st.write("Please upload a dataset to begin.")

# Section 1: Single Variable Charts
st.header("Single Variable Analysis")

# Histogram
st.subheader("Histogram")
st.write("Requirements: Select one numerical variable to view its distribution.")
col_hist = st.selectbox("Select a numerical variable for the histogram:", df.select_dtypes(include='number').columns)
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
st.write("Requirements: Select one numerical variable. Optionally, select a categorical variable for grouping.")
col_box = st.selectbox("Select a numerical variable for the boxplot:", df.select_dtypes(include='number').columns)
col_box_cat = st.selectbox("Select a categorical variable (optional):", [None] + list(df.select_dtypes(include='object').columns))
color_box = st.color_picker("Pick a color for the boxplot", "#1f77b4")

if col_box:
    fig, ax = plt.subplots()
    sns.boxplot(x=col_box_cat, y=col_box, data=df, color=color_box, ax=ax) if col_box_cat else sns.boxplot(y=df[col_box], color=color_box, ax=ax)
    ax.set_title(f"Boxplot of {col_box} grouped by {col_box_cat}" if col_box_cat else f"Boxplot of {col_box}")
    st.pyplot(fig)

# Section 2: Two Variable Charts
st.header("Two Variable Analysis")

# Scatter Plot
st.subheader("Scatter Plot")
st.write("Requirements: Select two numerical variables for plotting.")
col_scatter_x = st.selectbox("Select X-axis variable for scatter plot:", df.select_dtypes(include='number').columns)
col_scatter_y = st.selectbox("Select Y-axis variable for scatter plot:", df.select_dtypes(include='number').columns)
color_scatter = st.color_picker("Pick a color for scatter points", "#1f77b4")

if col_scatter_x and col_scatter_y and col_scatter_x != col_scatter_y:
    fig, ax = plt.subplots()
    sns.scatterplot(x=col_scatter_x, y=col_scatter_y, data=df, color=color_scatter, ax=ax)
    ax.set_title(f"Scatter Plot of {col_scatter_x} vs {col_scatter_y}")
    st.pyplot(fig)

# Line Plot
st.subheader("Line Plot")
st.write("Line plots are useful for time series or trends over a sequential index.")
col_line_y = st.selectbox("Select a numerical variable for line plot:", df.select_dtypes(include='number').columns, key="line_y")
time_col = st.selectbox("Select a time or index column (optional):", [None] + list(df.columns), key="time_col_line")
color_line = st.color_picker("Pick a color for the line", "#1f77b4")

if col_line_y:
    fig, ax = plt.subplots()
    sns.lineplot(x=df[time_col], y=df[col_line_y], color=color_line, ax=ax) if time_col else sns.lineplot(data=df[col_line_y], color=color_line, ax=ax)
    ax.set_title(f"Line Plot of {col_line_y} over {time_col}" if time_col else f"Line Plot of {col_line_y} over Index")
    st.pyplot(fig)

# Bar Plot
st.subheader("Bar Plot")
st.write("Requirements: Select a categorical variable and a numerical variable for summarization.")
col_bar_cat = st.selectbox("Select a categorical variable for bar plot:", df.select_dtypes(include='object').columns)
col_bar_num = st.selectbox("Select a numerical variable for bar plot:", df.select_dtypes(include='number').columns)
color_bar = st.color_picker("Pick a color for the bars", "#1f77b4")

if col_bar_cat and col_bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=col_bar_cat, y=col_bar_num, data=df, color=color_bar, ax=ax)
    ax.set_title(f"Bar Plot of {col_bar_num} by {col_bar_cat}")
    st.pyplot(fig)

# Density Plot
st.subheader("Density Plot")
st.write("Requirements: Select two numerical variables to view their joint density distribution.")
col_density_x = st.selectbox("Select X-axis variable for density plot:", df.select_dtypes(include='number').columns, key="density_x")
col_density_y = st.selectbox("Select Y-axis variable for density plot:", df.select_dtypes(include='number').columns, key="density_y")

if col_density_x and col_density_y and col_density_x != col_density_y:
    fig, ax = plt.subplots()
    sns.kdeplot(x=df[col_density_x], y=df[col_density_y], cmap="Blues", ax=ax, fill=True)
    ax.set_title(f"Density Plot of {col_density_x} and {col_density_y}")
    st.pyplot(fig)

# Hexbin Plot
st.subheader("Hexbin Plot")
st.write("Requirements: Select two numerical variables for a density-based scatter plot.")
col_hex_x = st.selectbox("Select X-axis variable for hexbin plot:", df.select_dtypes(include='number').columns, key="hex_x")
col_hex_y = st.selectbox("Select Y-axis variable for hexbin plot:", df.select_dtypes(include='number').columns, key="hex_y")
gridsize = st.slider("Hexbin grid size:", 10, 50, 30)

if col_hex_x and col_hex_y and col_hex_x != col_hex_y:
    fig, ax = plt.subplots()
    ax.hexbin(df[col_hex_x], df[col_hex_y], gridsize=gridsize, cmap="Blues")
    ax.set_title(f"Hexbin Plot of {col_hex_x} vs {col_hex_y}")
    st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
st.write("Requirements: Requires at least two numerical variables.")
if len(df.select_dtypes(include='number').columns) >= 2:
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.write("Not enough numerical variables for a correlation heatmap.")

# Section 3: Three Variable Chart - Pair Plot
st.header("Three Variable Analysis")

st.subheader("Pair Plot")
st.write("Requirements: Select a subset of numerical variables (at least two) and an optional categorical variable for hue.")
num_cols = st.multiselect("Select numerical variables for pair plot:", df.select_dtypes(include='number').columns)
hue_col = st.selectbox("Select a categorical variable for color coding (optional):", [None] + list(df.select_dtypes(include='object').columns))

if len(num_cols) >= 2:
    fig = sns.pairplot(df, vars=num_cols, hue=hue_col)
    st.pyplot(fig)
else:
    st.write("Please select at least two numerical variables for a pair plot.")
