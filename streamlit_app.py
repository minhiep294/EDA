import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to display guidelines for each chart
def display_guidelines():
    st.sidebar.header("Chart Requirements & Guidelines")
    st.sidebar.markdown("""
        - **Histogram**: Requires a single numerical variable.
        - **Boxplot**: Requires a numerical variable, optionally a categorical variable for grouping.
        - **Scatter Plot**: Requires two numerical variables; supports color, size aesthetics.
        - **Correlation Heatmap**: Requires at least two numerical variables.
        - **Bar Plot**: Requires one categorical and one numerical variable.
        - **Line Chart**: Supports two numerical variables or time series data.
    """)

# App title and file upload
st.title("Enhanced Data Exploration with Streamlit")
st.write("Upload a dataset to explore variables with various charts and aesthetics.")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    display_guidelines()
else:
    st.write("Please upload a dataset to begin.")

# Section 1: Single Variable Analysis
st.header("Single Variable Analysis")

# Histogram with Color Aesthetic
st.subheader("Histogram")
st.write("Select a numerical variable to view its distribution with optional color coding.")
col_hist = st.selectbox("Select numerical variable for histogram:", df.select_dtypes(include='number').columns)
hist_color = st.selectbox("Select a categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))
bins = st.slider("Number of bins:", 5, 50, 10)

if col_hist:
    fig, ax = plt.subplots()
    sns.histplot(df, x=col_hist, hue=hist_color, bins=bins, kde=True, ax=ax)
    ax.set_title(f"Histogram of {col_hist}")
    st.pyplot(fig)

# Boxplot with Size and Color Aesthetics
st.subheader("Boxplot")
st.write("Select a numerical variable and an optional categorical variable for grouping.")
col_box = st.selectbox("Select numerical variable for boxplot:", df.select_dtypes(include='number').columns)
col_box_cat = st.selectbox("Select categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns))

if col_box:
    fig, ax = plt.subplots()
    sns.boxplot(x=col_box_cat, y=col_box, data=df, ax=ax)
    ax.set_title(f"Boxplot of {col_box}" + (f" grouped by {col_box_cat}" if col_box_cat else ""))
    st.pyplot(fig)

# Section 2: Two Variable Analysis
st.header("Two Variable Analysis")

# Scatter Plot with Color and Size Aesthetics
st.subheader("Scatter Plot")
st.write("Select two numerical variables. Add color and size for enhanced aesthetics.")

scatter_x = st.selectbox("Select X-axis variable:", df.select_dtypes(include='number').columns)
scatter_y = st.selectbox("Select Y-axis variable:", df.select_dtypes(include='number').columns)
scatter_color = st.selectbox("Select categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))
scatter_size = st.selectbox("Select numerical variable for size (optional):", [None] + list(df.select_dtypes(include='number').columns))

if scatter_x and scatter_y:
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[scatter_x], y=df[scatter_y], hue=scatter_color, size=scatter_size, data=df, ax=ax)
    ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
    st.pyplot(fig)

# Line Chart with Color Aesthetic
st.subheader("Line Chart for Two Variables")
st.write("Select two numerical variables or use a time column for the x-axis.")

line_y1 = st.selectbox("Select first numerical variable for line chart:", df.select_dtypes(include='number').columns)
line_y2 = st.selectbox("Select second numerical variable for line chart:", df.select_dtypes(include='number').columns)
time_col_line = st.selectbox("Select a time or index column for x-axis (optional):", [None] + list(df.columns))

if line_y1 and line_y2 and line_y1 != line_y2:
    fig, ax = plt.subplots()
    if time_col_line:
        sns.lineplot(x=df[time_col_line], y=df[line_y1], ax=ax, label=line_y1)
        sns.lineplot(x=df[time_col_line], y=df[line_y2], ax=ax, label=line_y2)
        ax.set_title(f"Line Chart of {line_y1} and {line_y2} over {time_col_line}")
        ax.set_xlabel(time_col_line)
    else:
        sns.lineplot(data=df[[line_y1, line_y2]], ax=ax)
        ax.set_title(f"Line Chart of {line_y1} and {line_y2} over Index")
        ax.set_xlabel("Index")
    ax.set_ylabel("Values")
    st.pyplot(fig)

# Bar Chart with Color Aesthetic
st.subheader("Bar Chart")
st.write("Select a categorical and a numerical variable. Optionally, color by a second categorical variable.")

bar_cat = st.selectbox("Select categorical variable for bar chart:", df.select_dtypes(include='object').columns)
bar_num = st.selectbox("Select numerical variable for bar chart:", df.select_dtypes(include='number').columns)
bar_color = st.selectbox("Select a second categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))

if bar_cat and bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=bar_cat, y=bar_num, hue=bar_color, data=df, ax=ax)
    ax.set_title(f"Bar Chart of {bar_num} by {bar_cat}" + (f" with color by {bar_color}" if bar_color else ""))
    st.pyplot(fig)

# Density Plot with Color Aesthetic
st.subheader("Density Plot")
st.write("Select two numerical variables to view their joint density distribution.")

density_x = st.selectbox("Select X-axis variable for density plot:", df.select_dtypes(include='number').columns)
density_y = st.selectbox("Select Y-axis variable for density plot:", df.select_dtypes(include='number').columns)
density_color = st.selectbox("Select categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))

if density_x and density_y:
    fig, ax = plt.subplots()
    sns.kdeplot(x=df[density_x], y=df[density_y], hue=density_color, ax=ax, fill=True)
    ax.set_title(f"Density Plot of {density_x} and {density_y}" + (f" colored by {density_color}" if density_color else ""))
    st.pyplot(fig)

# Section 3: Three Variable Analysis
st.header("Three Variable Analysis")

# Pair Plot with Color Aesthetic
st.subheader("Pair Plot")
st.write("Select numerical variables for a pair plot, with optional color by a categorical variable.")

pair_cols = st.multiselect("Select numerical variables for pair plot:", df.select_dtypes(include='number').columns)
pair_hue = st.selectbox("Select categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns))

if len(pair_cols) >= 2:
    fig = sns.pairplot(df, vars=pair_cols, hue=pair_hue)
    st.pyplot(fig)
else:
    st.write("Please select at least two numerical variables for a pair plot.")
