import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def downsample_data(df, sample_size=1000):
    """Return a sample of the dataframe if it exceeds the sample size."""
    return df.sample(sample_size) if len(df) > sample_size else df

@st.cache_data
def calculate_correlation(df):
    """Calculate correlation matrix for numerical columns."""
    return df.corr()

# Initialize the app and sidebar
st.title("EDA with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")

# File upload section
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Identify numerical and categorical columns
    num_list = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_list = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

    # Display numerical and categorical columns
    st.write("Numerical Columns:", num_list)
    st.write("Categorical Columns:", cat_list)

    # Define tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning & Descriptive Stats", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

    with tab1:
        # Data Cleaning & Descriptive Stats
        st.header("Data Cleaning & Descriptive Stats")
        
        # Handle missing values
        st.subheader("Handle Missing Values")
        missing_option = st.radio("Choose a method to handle missing values:", ("Impute with Mean", "Remove Rows with Missing Data", "Leave as is"))
        if missing_option == "Impute with Mean":
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_option == "Remove Rows with Missing Data":
            df.dropna(inplace=True)

        # Display Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe(include='all'))

    with tab2:
        # Univariate Analysis
        st.header("Univariate Analysis")

        # Numerical Data Visualization
        st.subheader("Numerical Data")
        num_col = st.selectbox("Select a numerical variable:", num_list)
        if num_col:
            # Downsample data if necessary
            sampled_df = downsample_data(df)
            
            # Histogram
            fig, ax = plt.subplots()
            sns.histplot(sampled_df[num_col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {num_col}")
            st.pyplot(fig)

            # Box Plot
            fig, ax = plt.subplots()
            sns.boxplot(x=sampled_df[num_col], ax=ax)
            ax.set_title(f"Box Plot of {num_col}")
            st.pyplot(fig)

    with tab3:
        # Bivariate Analysis
        st.header("Bivariate Analysis")

        # Scatter Plot
        st.subheader("Scatter Plot")
        scatter_x = st.selectbox("Select X-axis variable:", num_list, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis variable:", num_list, key="scatter_y")

        if scatter_x and scatter_y:
            # Downsample for large data
            sampled_df = downsample_data(df)
            fig, ax = plt.subplots()
            sns.scatterplot(x=sampled_df[scatter_x], y=sampled_df[scatter_y], data=sampled_df, ax=ax)
            ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
            st.pyplot(fig)

        # Correlation Coefficient
        st.subheader("Correlation Coefficient")
        num_x_corr = st.selectbox("Select first numerical variable:", num_list, key="num_x_corr")
        num_y_corr = st.selectbox("Select second numerical variable:", num_list, key="num_y_corr")
        if num_x_corr and num_y_corr:
            corr_value = df[num_x_corr].corr(df[num_y_corr])
            st.write(f"Correlation between {num_x_corr} and {num_y_corr}: {corr_value:.2f}")

    with tab4:
        # Multivariate Analysis
        st.header("Multivariate Analysis")

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        if len(num_list) > 1:
            corr_matrix = calculate_correlation(df[num_list])
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)

        # Pair Plot (downsampled if large data)
        st.subheader("Pair Plot")
        if len(num_list) > 1:
            sampled_df = downsample_data(df[num_list])
            sns.pairplot(sampled_df)
            st.pyplot()

        # Box Plot for Numerical vs. Categorical
        st.subheader("Box Plot for Numerical and Categorical Variables")
        num_for_box = st.selectbox("Select a numerical variable for box plot:", num_list, key="num_for_box")
        cat_for_box = st.selectbox("Select a categorical variable for grouping in box plot:", cat_list, key="cat_for_box")
        if num_for_box and cat_for_box:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat_for_box, y=num_for_box, data=sampled_df, ax=ax)
            ax.set_title(f"Box Plot of {num_for_box} grouped by {cat_for_box}")
            st.pyplot(fig)
