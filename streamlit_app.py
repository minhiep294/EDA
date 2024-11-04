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

# Section 1: One Variable Analysis
st.header("One Variable Analysis")

# Amounts: Bar Chart and Dot Plot
st.subheader("Visualizing Amounts (One Variable)")

# Bar Chart
st.write("### Bar Chart")
st.write("Useful for visualizing amounts across categories.")
bar_cat = st.selectbox("Select a categorical variable for bar chart:", df.select_dtypes(include='object').columns, key="bar_cat")
bar_num = st.selectbox("Select a numerical variable for bar chart:", df.select_dtypes(include='number').columns, key="bar_num")

if bar_cat and bar_num:
    fig, ax = plt.subplots()
    sns.barplot(x=bar_cat, y=bar_num, data=df, ax=ax)
    ax.set_title(f"Bar Chart of {bar_num} by {bar_cat}")
    st.pyplot(fig)

# Dot Plot
st.write("### Dot Plot")
st.write("Alternative to bar chart, showing data points instead of bars.")
if bar_cat and bar_num:
    fig, ax = plt.subplots()
    sns.stripplot(x=bar_num, y=bar_cat, data=df, ax=ax, jitter=False)
    ax.set_title(f"Dot Plot of {bar_num} by {bar_cat}")
    st.pyplot(fig)

# Distributions: Histogram, Density Plot, Boxplot, Violin Plot
st.subheader("Visualizing Distributions (One Variable)")

# Histogram
st.write("### Histogram")
hist_col = st.selectbox("Select numerical variable for histogram:", df.select_dtypes(include='number').columns, key="hist_col")
hist_color = st.selectbox("Select a categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns), key="hist_color")
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

# Violin Plot
st.write("### Violin Plot")
if box_col:
    fig, ax = plt.subplots()
    sns.violinplot(x=box_cat, y=box_col, data=df, ax=ax)
    ax.set_title(f"Violin Plot of {box_col} grouped by {box_cat}" if box_cat else f"Violin Plot of {box_col}")
    st.pyplot(fig)

# Section 2: Two Variable Analysis - X-Y Relationships and Proportions
st.header("Two Variable Analysis")

# X-Y Relationships: Scatter Plot, Line Chart
st.subheader("Visualizing X-Y Relationships (Two Variables)")

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

# Line Chart
st.write("### Line Chart")
line_x = st.selectbox("Select time or index for X-axis:", [None] + list(df.columns), key="line_x")
line_y1 = st.selectbox("Select first numerical variable:", df.select_dtypes(include='number').columns, key="line_y1")
line_y2 = st.selectbox("Select second numerical variable (optional):", [None] + list(df.select_dtypes(include='number').columns), key="line_y2")

if line_x and line_y1:
    fig, ax = plt.subplots()
    sns.lineplot(x=df[line_x], y=df[line_y1], ax=ax, label=line_y1)
    if line_y2:
        sns.lineplot(x=df[line_x], y=df[line_y2], ax=ax, label=line_y2)
    ax.set_title(f"Line Chart of {line_y1}" + (f" and {line_y2}" if line_y2 else "") + f" over {line_x}")
    st.pyplot(fig)

# Proportions: Stacked Bar Chart
st.subheader("Visualizing Proportions (Two Variables)")

# Stacked Bar Chart
st.write("### Stacked Bar Chart")
stack_cat1 = st.selectbox("Select first categorical variable for stacking:", df.select_dtypes(include='object').columns, key="stack_cat1")
stack_cat2 = st.selectbox("Select second categorical variable:", df.select_dtypes(include='object').columns, key="stack_cat2")

if stack_cat1 and stack_cat2:
    stacked_data = df.groupby([stack_cat1, stack_cat2]).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    stacked_data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"Stacked Bar Chart of {stack_cat1} by {stack_cat2}")
    st.pyplot(fig)

# Section 3: Three Variable Analysis - Relationships with Multiple Variables and Uncertainty
st.header("Three Variable Analysis")

# Bubble Chart
st.subheader("Bubble Chart for Three Variables")
st.write("A bubble chart is useful for visualizing relationships between three numerical variables, where the size of the bubble represents a third variable.")

bubble_x = st.selectbox("Select X-axis variable for bubble chart:", df.select_dtypes(include='number').columns, key="bubble_x")
bubble_y = st.selectbox("Select Y-axis variable for bubble chart:", df.select_dtypes(include='number').columns, key="bubble_y")
bubble_size = st.selectbox("Select variable for bubble size:", df.select_dtypes(include='number').columns, key="bubble_size")
bubble_color = st.selectbox("Select categorical variable for bubble color (optional):", [None] + list(df.select_dtypes(include='object').columns), key="bubble_color")

if bubble_x and bubble_y and bubble_size:
    fig, ax = plt.subplots()
    sns.scatterplot(x=bubble_x, y=bubble_y, size=bubble_size, hue=bubble_color, sizes=(20, 200), data=df, ax=ax, legend="brief")
    ax.set_title(f"Bubble Chart of {bubble_x} vs {bubble_y} with bubble size based on {bubble_size}")
    st.pyplot(fig)

# 3D Scatter Plot
st.subheader("3D Scatter Plot for Three Variables")
st.write("A 3D scatter plot to visualize relationships among three numerical variables.")

scatter_3d_x = st.selectbox("Select X-axis variable for 3D scatter plot:", df.select_dtypes(include='number').columns, key="scatter_3d_x")
scatter_3d_y = st.selectbox("Select Y-axis variable for 3D scatter plot:", df.select_dtypes(include='number').columns, key="scatter_3d_y")
scatter_3d_z = st.selectbox("Select Z-axis variable for 3D scatter plot:", df.select_dtypes(include='number').columns, key="scatter_3d_z")
scatter_3d_color = st.selectbox("Select categorical variable for color (optional):", [None] + list(df.select_dtypes(include='object').columns), key="scatter_3d_color")

if scatter_3d_x and scatter_3d_y and scatter_3d_z:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df[scatter_3d_x], df[scatter_3d_y], df[scatter_3d_z], c=df[scatter_3d_color].astype('category').cat.codes if scatter_3d_color else None, cmap='viridis', s=50)
    ax.set_xlabel(scatter_3d_x)
    ax.set_ylabel(scatter_3d_y)
    ax.set_zlabel(scatter_3d_z)
    ax.set_title(f"3D Scatter Plot of {scatter_3d_x}, {scatter_3d_y}, and {scatter_3d_z}")
    if scatter_3d_color:
        plt.colorbar(sc, ax=ax, label=scatter_3d_color)
    st.pyplot(fig)

# Facet Grid (Small Multiples)
st.subheader("Facet Grid (Small Multiples)")
st.write("Facet Grid is useful for visualizing distributions of numerical variables across levels of a categorical variable.")

facet_col = st.selectbox("Select a categorical variable for facets:", df.select_dtypes(include='object').columns, key="facet_col")
facet_num_x = st.selectbox("Select numerical variable for X-axis:", df.select_dtypes(include='number').columns, key="facet_num_x")
facet_num_y = st.selectbox("Select numerical variable for Y-axis (optional):", [None] + list(df.select_dtypes(include='number').columns), key="facet_num_y")

if facet_col and facet_num_x:
    if facet_num_y:
        # Scatter plot in Facet Grid
        g = sns.FacetGrid(df, col=facet_col, height=4, aspect=1)
        g.map_dataframe(sns.scatterplot, x=facet_num_x, y=facet_num_y)
        g.set_axis_labels(f"{facet_num_x}", f"{facet_num_y}")
    else:
        # Histogram in Facet Grid
        g = sns.FacetGrid(df, col=facet_col, height=4, aspect=1)
        g.map_dataframe(sns.histplot, x=facet_num_x, kde=True)
        g.set_axis_labels(f"{facet_num_x}", "Density")

    g.add_legend()
    g.fig.suptitle(f"Facet Grid of {facet_num_x} across {facet_col} categories", y=1.05)
    st.pyplot(g)
