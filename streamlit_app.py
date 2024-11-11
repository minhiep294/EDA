import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the app and sidebar
st.title("EDA with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")

# File upload section
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Identify numerical and categorical columns
    num_list = []
    cat_list = []
    for column in df:
        if pd.api.types.is_numeric_dtype(df[column]):
            num_list.append(column)
        elif pd.api.types.is_string_dtype(df[column]):
            cat_list.append(column)
    st.write("Numerical Columns:", num_list)
    st.write("Categorical Columns:", cat_list)

    # Define tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning & Descriptive Stats", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

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
            try:
                if col_type == "Integer":
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
                elif col_type == "Float":
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif col_type == "String":
                    df[col] = df[col].astype(str)
                elif col_type == "DateTime":
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.write(f"Could not convert {col} to {col_type}: {e}")
        st.write("Data Cleaning Complete.")
        st.write(df.head())

        # Section 2: Descriptive Statistics
        st.header("2. Descriptive Statistics")
        st.subheader("Central Tendency & Dispersion")
        st.write(df.describe(include='all'))

        if st.checkbox("Show Mode"):
            st.write(df.mode().iloc[0])
            
    with tab2:
        # Univariate Analysis
        st.header("Univariate Analysis")
    
        # Numerical Data Visualization
        st.subheader("Numerical Data Visualization")
        num_col = st.selectbox("Select a numerical variable:", num_list)
        if num_col:
            # Histogram
            fig, ax = plt.subplots()
            sns.histplot(df[num_col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {num_col}")
            st.pyplot(fig)
    
            # Box Plot
            fig, ax = plt.subplots()
            sns.boxplot(x=df[num_col], ax=ax)
            ax.set_title(f"Box Plot of {num_col}")
            st.pyplot(fig)
    
            # Density Plot
            fig, ax = plt.subplots()
            sns.kdeplot(df[num_col], fill=True, ax=ax)
            ax.set_title(f"Density Plot of {num_col}")
            st.pyplot(fig)
    
        # Categorical Data Visualization
        st.subheader("Categorical Data Visualization")
        cat_col = st.selectbox("Select a categorical variable:", cat_list)
        if cat_col:
            # Count Plot
            fig, ax = plt.subplots()
            sns.countplot(x=df[cat_col], ax=ax)
            ax.set_title(f"Count Plot of {cat_col}")
            st.pyplot(fig)
    
            # Bar Chart
            fig, ax = plt.subplots()
            sns.barplot(x=df[cat_col].value_counts().index, y=df[cat_col].value_counts().values, ax=ax)
            ax.set_title(f"Bar Chart of {cat_col}")
            st.pyplot(fig)
    
            # Pie Plot
            fig, ax = plt.subplots()
            df[cat_col].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_ylabel('')
            ax.set_title(f"Pie Plot of {cat_col}")
            st.pyplot(fig)
    
    with tab3:
        # Bivariate Analysis
        st.header("Bivariate Analysis")

        # Numerical vs. Numerical
        st.subheader("Numerical vs. Numerical")
        
        # Pair Plot
        st.write("### Pair Plot")
        if len(df.select_dtypes(include='number').columns) > 1:
            pair_plot = sns.pairplot(df[num_list])
            st.pyplot(pair_plot.fig)
        
        # Scatter Plot with optional color and size
        st.write("### Scatter Plot")
        scatter_x = st.selectbox("Select X-axis variable:", num_list, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis variable:", num_list, key="scatter_y")
        scatter_color = st.selectbox("Select a categorical variable for color (optional):", [""] + cat_list, key="scatter_color")
        scatter_size = st.selectbox("Select numerical variable for size (optional):", [""] + num_list, key="scatter_size")
        
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[scatter_x], y=df[scatter_y], hue=df[scatter_color] if scatter_color else None, 
                            size=df[scatter_size] if scatter_size else None, data=df, ax=ax)
            ax.set_title(f"Scatter Plot of {scatter_x} vs {scatter_y}")
            st.pyplot(fig)
        
        # Correlation Coefficient
        st.write("### Correlation Coefficient")
        num_x_corr = st.selectbox("Select first numerical variable:", num_list, key="num_x_corr")
        num_y_corr = st.selectbox("Select second numerical variable:", num_list, key="num_y_corr")
        
        if num_x_corr and num_y_corr:
            corr_value = df[num_x_corr].corr(df[num_y_corr])
            st.write(f"Correlation between {num_x_corr} and {num_y_corr}: {corr_value:.2f}")
        
        # Categorical vs. Numerical
        st.subheader("Categorical vs. Numerical")
        
        # Bar Plot
        cat_for_bar = st.selectbox("Select a categorical variable for bar plot:", cat_list, key="cat_for_bar")
        num_for_bar = st.selectbox("Select a numerical variable for bar plot:", num_list, key="num_for_bar")
        
        if cat_for_bar and num_for_bar:
            fig, ax = plt.subplots()
            sns.barplot(x=cat_for_bar, y=num_for_bar, data=df, ax=ax)
            ax.set_title(f"Bar Plot of {num_for_bar} by {cat_for_bar}")
            st.pyplot(fig)
        
    with tab4:
        # Multivariate Analysis
        st.header("Multivariate Analysis")

        # Correlation Matrix
        st.write("### Correlation Matrix")
        if len(num_list) > 1:
            corr_matrix = df[num_list].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix for Numerical Variables")
            st.pyplot(fig)
        
        # Pair Plot with Hue
        st.write("### Pair Plot with Hue for Numerical and Categorical Variables")
        hue_cat = st.selectbox("Select a categorical variable for hue:", cat_list, key="hue_cat_for_pairplot")
        if hue_cat:
            pair_plot_hue = sns.pairplot(df, vars=num_list, hue=hue_cat)
            st.pyplot(pair_plot_hue.fig)
