import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to display visualization guidance
def display_visualization_guide():
    st.sidebar.header("Directory of Visualizations")
    st.sidebar.markdown("""
    ### 5.1 Amounts
    - **Bar Chart**: Shows values across categories. Can be grouped or stacked.
    - **Dot Plot**: Alternative to bar charts, showing points at end values.
    
    ### 5.2 Distributions
    - **Histogram**: Basic distribution of a numerical variable.
    - **Density Plot**: Smoothed curve of data distribution.
    - **Boxplot**: Summarizes distribution with quartiles and outliers.
    - **Violin Plot**: Combination of boxplot and density plot.
    
    ### 5.3 Proportions
    - **Pie Chart**: Displays parts of a whole.
    - **Stacked Bar Chart**: Shows proportions across multiple categories.
    
    ### 5.4 X–Y Relationships
    - **Scatter Plot**: Shows relationship between two numerical variables.
    - **Line Chart**: Displays trends over time or other ordered data.
    - **Bubble Chart**: Adds a third variable as dot size in scatter plot.
    
    ### 5.5 Uncertainty
    - **Confidence Interval Plot**: Shows range of likely values.
    - **Error Bars**: Represents uncertainty in measurements.
    """)

# App title and file upload
st.title("Enhanced EDA with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    display_visualization_guide()
else:
    st.write("Please upload a dataset to begin.")

# Section 1: Single Variable Analysis
st.header("Single Variable Analysis")

# Amounts - Bar Chart and Dot Plot
st.subheader("Visualizing Amounts")

# Bar Chart
st.write("### Bar Chart")
st.write("Visualize amounts across categories. Use grouping or stacking for multiple categories.")
bar_cat = st.selectbox("Select categorical variable for bar chart:", df.select_dtypes(include='object').columns, key="bar_cat")
bar_num = st.selectbox("Select numerical variable for bar chart:", df.select_dtypes(include='number').columns, key="bar_num")
bar_color = st.selectbox("Select another categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns), key="bar_color")

if bar_cat and bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=bar_cat, y=bar_num, hue=bar_color, data=df, ax=ax)
    ax.set_title(f"Bar Chart of {bar_num} by {bar_cat}")
    st.pyplot(fig)

# Dot Plot (Alternative to Bar Chart)
st.write("### Dot Plot")
st.write("Alternative to bar chart, with points at the end value instead of bars.")
if bar_cat and bar_num:
    fig, ax = plt.subplots()
    sns.stripplot(x=bar_num, y=bar_cat, data=df, ax=ax, jitter=False)
    ax.set_title(f"Dot Plot of {bar_num} by {bar_cat}")
    st.pyplot(fig)

# Distributions - Histogram, Density Plot, and Boxplot
st.subheader("Visualizing Distributions")

# Histogram
st.write("### Histogram with Density")
hist_col = st.selectbox("Select numerical variable for histogram:", df.select_dtypes(include='number').columns, key="hist_col")
hist_color = st.selectbox("Select categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns), key="hist_color")
bins = st.slider("Number of bins:", 5, 50, 10, key="bins")

if hist_col:
    fig, ax = plt.subplots()
    sns.histplot(df, x=hist_col, hue=hist_color, bins=bins, kde=True, ax=ax)
    ax.set_title(f"Histogram of {hist_col}")
    st.pyplot(fig)

# Density Plot
st.write("### Density Plot")
if hist_col:
    fig, ax = plt.subplots()
    sns.kdeplot(df[hist_col], hue=hist_color, fill=True, ax=ax)
    ax.set_title(f"Density Plot of {hist_col}")
    st.pyplot(fig)

# Boxplot
st.write("### Boxplot")
box_col = st.selectbox("Select numerical variable for boxplot:", df.select_dtypes(include='number').columns, key="box_col")
box_cat = st.selectbox("Select categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns), key="box_cat")

if box_col:
    fig, ax = plt.subplots()
    sns.boxplot(x=box_cat, y=box_col, data=df, ax=ax)
    ax.set_title(f"Boxplot of {box_col} grouped by {box_cat}" if box_cat else f"Boxplot of {box_col}")
    st.pyplot(fig)

# Section 2: Two Variable Analysis - X-Y Relationships
st.header("Two Variable Analysis")
st.subheader("Visualizing X-Y Relationships")

# Scatter Plot
st.write("### Scatter Plot")
scatter_x = st.selectbox("Select X-axis variable:", df.select_dtypes(include='number').columns, key="scatter_x")
scatter_y = st.selectbox("Select Y-axis variable:", df.select_dtypes(include='number').columns, key="scatter_y")
scatter_color = st.selectbox("Select a categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns), key="scatter_color")
scatter_size = st.selectbox("Select numerical variable for size (optional):", [None] + list(df.select_dtypes(include='number').columns), key="scatter_size")

if scatter_x and scatter_y:
    fig, ax = plt.subplots()
    sns.scatterplot(x=scatter_x, y=scatter_y, hue=scatter_color, size=scatter_size, data=df, ax=ax)
    ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
    st.pyplot(fig)

# Line Chart for Time Series or Trend Analysis
st.write("### Line Chart")
line_col_x = st.selectbox("Select X-axis (time or index):", [None] + list(df.columns), key="line_col_x")
line_y1 = st.selectbox("Select first numerical variable:", df.select_dtypes(include='number').columns, key="line_y1")
line_y2 = st.selectbox("Select second numerical variable (optional):", [None] + list(df.select_dtypes(include='number').columns), key="line_y2")

if line_col_x and line_y1:
    fig, ax = plt.subplots()
    sns.lineplot(x=df[line_col_x], y=df[line_y1], ax=ax, label=line_y1)
    if line_y2:
        sns.lineplot(x=df[line_col_x], y=df[line_y2], ax=ax, label=line_y2)
    ax.set_title(f"Line Chart of {line_y1}" + (f" and {line_y2}" if line_y2 else "") + f" over {line_col_x}")
    st.pyplot(fig)

# Section 3: Three Variable Analysis - Pair Plot
st.header("Three Variable Analysis")

# Pair Plot
st.write("### Pair Plot for Three Variables")
pair_cols = st.multiselect("Select numerical variables for pair plot:", df.select_dtypes(include='number').columns, key="pair_cols")
pair_hue = st.selectbox("Select categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns), key="pair_hue")

if len(pair_cols) >= 2:
    fig = sns.pairplot(df, vars=pair_cols, hue=pair_hue)
    st.pyplot(fig)

# Uncertainty - Confidence Interval Plot
st.subheader("Visualizing Uncertainty")

# Confidence Interval Plot
st.write("### Confidence Interval Plot")
conf_num = st.selectbox("Select numerical variable for CI plot:", df.select_dtypes(include='number').columns, key="conf_num")
conf_cat = st.selectbox("Select categorical variable for grouping (optional):", [None] + list(df.select_dtypes(include='object').columns), key="conf_cat")

if conf_num:
    fig, ax = plt.subplots()
    sns.pointplot(x=conf_cat, y=conf_num, data=df, ci='sd', ax=ax)
    ax.set_title(f"Confidence Interval Plot of {conf_num}" + (f" grouped by {conf_cat}" if conf_cat else ""))
    st.pyplot(fig)
