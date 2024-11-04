import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helper functions for charts with brief guidelines
def display_guidelines():
    st.sidebar.header("Chart Requirements & Guidelines")
    st.sidebar.markdown("""
        - **Histogram:** Requires a single numerical variable.
        - **Boxplot:** Requires a single numerical variable, and optionally, a categorical variable for grouping.
        - **Scatter Plot:** Requires two numerical variables.
        - **Correlation Heatmap:** Requires at least two numerical variables.
        - **Bar Plot:** Requires one categorical and one numerical variable.
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

if col_hist:
    fig, ax = plt.subplots()
    sns.histplot(df[col_hist], bins=bins, kde=True, ax=ax)
    ax.set_title(f"Histogram of {col_hist}")
    st.pyplot(fig)

# Boxplot
st.subheader("Boxplot")
st.write("Requirements: Select one numerical variable. Optionally, select a categorical variable for grouping.")
col_box = st.selectbox("Select a numerical variable for the boxplot:", df.select_dtypes(include='number').columns)
col_box_cat = st.selectbox("Select a categorical variable (optional):", [None] + list(df.select_dtypes(include='object').columns))

if col_box:
    fig, ax = plt.subplots()
    if col_box_cat:
        sns.boxplot(x=col_box_cat, y=col_box, data=df, ax=ax)
        ax.set_title(f"Boxplot of {col_box} grouped by {col_box_cat}")
    else:
        sns.boxplot(y=df[col_box], ax=ax)
        ax.set_title(f"Boxplot of {col_box}")
    st.pyplot(fig)

# Section 2: Two Variable Charts
st.header("Two Variable Analysis")

#Line chart
st.subheader("Line Chart for Two Variables")
st.write("Requirements: Select two numerical variables. Optionally, select a time or index column for the x-axis.")

col_line_y1 = st.selectbox("Select the first numerical variable for the line chart:", df.select_dtypes(include='number').columns, key="line_y1")
col_line_y2 = st.selectbox("Select the second numerical variable for the line chart:", df.select_dtypes(include='number').columns, key="line_y2")
time_col_line = st.selectbox("Select an index or time column (optional):", [None] + list(df.columns), key="time_col_line")

if col_line_y1 and col_line_y2 and col_line_y1 != col_line_y2:
    fig, ax = plt.subplots()
    if time_col_line:
        sns.lineplot(x=df[time_col_line], y=df[col_line_y1], ax=ax, label=col_line_y1)
        sns.lineplot(x=df[time_col_line], y=df[col_line_y2], ax=ax, label=col_line_y2)
        ax.set_title(f"Line Chart of {col_line_y1} and {col_line_y2} over {time_col_line}")
        ax.set_xlabel(time_col_line)
    else:
        sns.lineplot(data=df[[col_line_y1, col_line_y2]], ax=ax)
        ax.set_title(f"Line Chart of {col_line_y1} and {col_line_y2} over Index")
        ax.set_xlabel("Index")
    ax.set_ylabel("Values")
    st.pyplot(fig)
# Bar Chart
st.subheader("Bar Chart")
st.write("Requirements: Select one categorical variable and one numerical variable.")

col_bar_cat = st.selectbox("Select a categorical variable for the bar chart:", df.select_dtypes(include='object').columns, key="bar_cat")
col_bar_num = st.selectbox("Select a numerical variable for the bar chart:", df.select_dtypes(include='number').columns, key="bar_num")

if col_bar_cat and col_bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=col_bar_cat, y=col_bar_num, data=df, ax=ax)
    ax.set_title(f"Bar Chart of {col_bar_num} by {col_bar_cat}")
    st.pyplot(fig)

# Stacked Bar Chart
st.subheader("Stacked Bar Chart")
st.write("Requirements: Select two categorical variables. The chart will display counts with one variable stacked over the other.")

col_stack_cat1 = st.selectbox("Select the first categorical variable for the stacked bar chart:", df.select_dtypes(include='object').columns, key="stack_cat1")
col_stack_cat2 = st.selectbox("Select the second categorical variable for the stacked bar chart:", df.select_dtypes(include='object').columns, key="stack_cat2")

if col_stack_cat1 and col_stack_cat2 and col_stack_cat1 != col_stack_cat2:
    stacked_data = df.groupby([col_stack_cat1, col_stack_cat2]).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    stacked_data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"Stacked Bar Chart of {col_stack_cat1} by {col_stack_cat2}")
    ax.set_xlabel(col_stack_cat1)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Count Plot
st.subheader("Count Plot")
st.write("Requirements: Select one categorical variable. Optionally, select a second categorical variable for hue.")

col_count_cat = st.selectbox("Select a categorical variable for the count plot:", df.select_dtypes(include='object').columns, key="count_cat")
col_count_hue = st.selectbox("Select a second categorical variable (optional):", [None] + list(df.select_dtypes(include='object').columns), key="count_hue")

if col_count_cat:
    fig, ax = plt.subplots()
    sns.countplot(x=col_count_cat, hue=col_count_hue, data=df, ax=ax)
    ax.set_title(f"Count Plot of {col_count_cat}" + (f" by {col_count_hue}" if col_count_hue else ""))
    st.pyplot(fig)

# Density Plot
st.subheader("Density Plot")
st.write("Requirements: Select two numerical variables to view their joint density distribution.")

col_density_x = st.selectbox("Select X-axis variable for density plot:", df.select_dtypes(include='number').columns, key="density_x")
col_density_y = st.selectbox("Select Y-axis variable for density plot:", df.select_dtypes(include='number').columns, key="density_y")

if col_density_x and col_density_y and col_density_x != col_density_y:
    fig, ax = plt.subplots()
    sns.kdeplot(x=df[col_density_x], y=df[col_density_y], ax=ax, fill=True, cmap="Blues")
    ax.set_title(f"Density Plot of {col_density_x} and {col_density_y}")
    st.pyplot(fig)

# Section 3: Three Variable Chart (for advanced cases)
st.header("Three Variable Analysis")

# Pair Plot
st.subheader("Pair Plot")
st.write("Requirements: Select a subset of numerical variables (at least two) and an optional categorical variable for hue.")
num_cols = st.multiselect("Select numerical variables for pair plot:", df.select_dtypes(include='number').columns)
hue_col = st.selectbox("Select a categorical variable for color coding (optional):", [None] + list(df.select_dtypes(include='object').columns))

if len(num_cols) >= 2:
    fig = sns.pairplot(df, vars=num_cols, hue=hue_col)
    st.pyplot(fig)
else:
    st.write("Please select at least two numerical variables for a pair plot.")
