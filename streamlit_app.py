import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Helper function to display visualization guidance in the sidebar
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

# Initialize the app and sidebar
st.title("EDA with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")
display_visualization_guide()

# File upload section
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Define tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning & Descriptive Stats", "One Variable Analysis", "Two Variable Analysis", "Three Variable Analysis"])

    with tab1:
        # Section 1: Data Cleaning
        st.header("1. Data Cleaning")
        st.subheader("Handle Missing Values")
        missing_option = st.radio("Choose a method to handle missing values:", ("Impute with Mean", "Remove Rows with Missing Data", "Leave as is"))
        if missing_option == "Impute with Mean":
            # Only fill missing values for numeric columns
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_option == "Remove Rows with Missing Data":
            df.dropna(inplace=True)
        
        st.subheader("Remove Duplicates")
        if st.button("Remove Duplicate Rows"):
            before = df.shape[0]
            df.drop_duplicates(inplace=True)
            after = df.shape[0]
            st.write(f"Removed {before - after} duplicate rows")

        st.subheader("Correct Data Types")
        for col in df.columns:
            col_type = st.selectbox(f"Select data type for {col}", ("Automatic", "Integer", "Float", "String", "DateTime"), index=0)
            if col_type == "Integer":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
            elif col_type == "Float":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col_type == "String":
                df[col] = df[col].astype(str)
            elif col_type == "DateTime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
        st.write("Data Cleaning Complete.")
        st.write(df.head())

        # Section 2: Descriptive Statistics
        st.header("2. Descriptive Statistics")
        st.subheader("Central Tendency & Dispersion")
        st.write(df.describe(include='all'))

        if st.checkbox("Show Mode"):
            st.write(df.mode().iloc[0])
    
    with tab2:
        # Dynamic visualization options for One Variable Analysis
        st.header("One Variable Analysis")
        st.subheader("Visualizing Amounts (One Variable)")

        # Bar Chart for categorical vs numerical
        bar_cat = st.selectbox("Select categorical variable for bar chart:", df.select_dtypes(include='object').columns, key="bar_cat")
        bar_num = st.selectbox("Select numerical variable for bar chart:", df.select_dtypes(include='number').columns, key="bar_num")
        if bar_cat and bar_num:
            fig, ax = plt.subplots()
            sns.barplot(x=bar_cat, y=bar_num, data=df, ax=ax)
            ax.set_title(f"Bar Chart of {bar_num} by {bar_cat}")
            st.pyplot(fig)

        # Histogram for numerical distributions
        st.write("### Histogram")
        hist_col = st.selectbox("Select numerical variable for histogram:", df.select_dtypes(include='number').columns, key="hist_col")
        if hist_col:
            fig, ax = plt.subplots()
            sns.histplot(df[hist_col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {hist_col}")
            st.pyplot(fig)

        # Boxplot
        st.write("### Boxplot")
        box_num = st.selectbox("Select numerical variable for boxplot:", df.select_dtypes(include='number').columns, key="box_num")
        if box_num:
            fig, ax = plt.subplots()
            sns.boxplot(y=box_num, data=df, ax=ax)
            ax.set_title(f"Boxplot of {box_num}")
            st.pyplot(fig)

    with tab3:
        # Section for Two Variable Analysis
        st.header("Two Variable Analysis")
        st.subheader("Visualizing Relationships (Two Variables)")

        # Scatter Plot for numerical relationships
        scatter_x = st.selectbox("Select X-axis variable:", df.select_dtypes(include='number').columns, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis variable:", df.select_dtypes(include='number').columns, key="scatter_y")
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            sns.scatterplot(x=scatter_x, y=scatter_y, data=df, ax=ax)
            ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
            st.pyplot(fig)

        # Line Chart for time series
        line_x = st.selectbox("Select variable for X-axis (time/index):", df.columns, key="line_x")
        line_y = st.selectbox("Select variable for Y-axis:", df.select_dtypes(include='number').columns, key="line_y")
        if line_x and line_y:
            fig, ax = plt.subplots()
            sns.lineplot(x=line_x, y=line_y, data=df, ax=ax)
            ax.set_title(f"Line Chart of {line_y} over {line_x}")
            st.pyplot(fig)

    with tab4:
        # Section for Three Variable Analysis
        st.header("Three Variable Analysis")
        st.subheader("3D Scatter Plot (Three Variables)")

        # 3D Scatter Plot for three numerical variables
        scatter_3d_x = st.selectbox("Select X-axis variable:", df.select_dtypes(include='number').columns, key="scatter_3d_x")
        scatter_3d_y = st.selectbox("Select Y-axis variable:", df.select_dtypes(include='number').columns, key="scatter_3d_y")
        scatter_3d_z = st.selectbox("Select Z-axis variable:", df.select_dtypes(include='number').columns, key="scatter_3d_z")
        if scatter_3d_x and scatter_3d_y and scatter_3d_z:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[scatter_3d_x], df[scatter_3d_y], df[scatter_3d_z])
            ax.set_xlabel(scatter_3d_x)
            ax.set_ylabel(scatter_3d_y)
            ax.set_zlabel(scatter_3d_z)
            ax.set_title(f"3D Scatter Plot of {scatter_3d_x}, {scatter_3d_y}, and {scatter_3d_z}")
            st.pyplot(fig)

else:
    st.write("Please upload a dataset to begin.")
