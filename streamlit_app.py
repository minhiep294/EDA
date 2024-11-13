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

    # tab1 content remains the same...

    with tab2:
        # Univariate Analysis
        st.header("Univariate Analysis")
        
        # Numerical Data Visualization
        st.subheader("Numerical Data Visualization")
        num_col = st.selectbox("Select a numerical variable:", num_list)
        if num_col:
            chart_type = st.selectbox("Select chart type:", ["Histogram", "Box Plot", "Density Plot"])
            
            # Draw only the selected chart type
            fig, ax = plt.subplots()
            if chart_type == "Histogram":
                sns.histplot(df[num_col], kde=True, ax=ax)
                ax.set_title(f"Histogram of {num_col}")
            elif chart_type == "Box Plot":
                sns.boxplot(x=df[num_col], ax=ax)
                ax.set_title(f"Box Plot of {num_col}")
            elif chart_type == "Density Plot":
                sns.kdeplot(df[num_col], fill=True, ax=ax)
                ax.set_title(f"Density Plot of {num_col}")
            st.pyplot(fig)

        # Categorical Data Visualization
        st.subheader("Categorical Data Visualization")
        cat_col = st.selectbox("Select a categorical variable:", cat_list)
        if cat_col:
            cat_chart_type = st.selectbox("Select chart type:", ["Count Plot", "Bar Chart", "Pie Plot"])
            
            # Draw only the selected chart type
            fig, ax = plt.subplots()
            if cat_chart_type == "Count Plot":
                sns.countplot(x=df[cat_col], ax=ax)
                ax.set_title(f"Count Plot of {cat_col}")
            elif cat_chart_type == "Bar Chart":
                sns.barplot(x=df[cat_col].value_counts().index, y=df[cat_col].value_counts().values, ax=ax)
                ax.set_title(f"Bar Chart of {cat_col}")
            elif cat_chart_type == "Pie Plot":
                df[cat_col].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_ylabel('')
                ax.set_title(f"Pie Plot of {cat_col}")
            st.pyplot(fig)

    with tab3:
        # Bivariate Analysis
        st.header("Bivariate Analysis")

        # Numerical vs. Numerical
        st.subheader("Numerical vs. Numerical")
        bivar_chart_type = st.selectbox("Select chart type:", ["Pair Plot", "Scatter Plot", "Correlation Coefficient"])
        
        # Pair Plot
        if bivar_chart_type == "Pair Plot" and len(num_list) > 1:
            pair_plot = sns.pairplot(df[num_list])
            st.pyplot(pair_plot.fig)
        
        # Scatter Plot with optional color and size
        elif bivar_chart_type == "Scatter Plot":
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
        elif bivar_chart_type == "Correlation Coefficient":
            num_x_corr = st.selectbox("Select first numerical variable:", num_list, key="num_x_corr")
            num_y_corr = st.selectbox("Select second numerical variable:", num_list, key="num_y_corr")
            
            if num_x_corr and num_y_corr:
                corr_value = df[num_x_corr].corr(df[num_y_corr])
                st.write(f"Correlation between {num_x_corr} and {num_y_corr}: {corr_value:.2f}")

    with tab4:
        # Multivariate Analysis
        st.header("Multivariate Analysis")

        multi_chart_type = st.selectbox("Select chart type:", ["Correlation Matrix", "Pair Plot with Hue", "Box Plot"])

        # Correlation Matrix
        if multi_chart_type == "Correlation Matrix" and len(num_list) > 1:
            corr_matrix = df[num_list].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix for Numerical Variables")
            st.pyplot(fig)

        # Pair Plot with Hue
        elif multi_chart_type == "Pair Plot with Hue" and len(num_list) > 1:
            hue_cat = st.selectbox("Select a categorical variable for hue:", cat_list, key="hue_cat_for_pairplot")
            if hue_cat:
                pair_plot_hue = sns.pairplot(df, vars=num_list, hue=hue_cat)
                st.pyplot(pair_plot_hue.fig)
        
        # Box Plot for Numerical vs. Categorical
        elif multi_chart_type == "Box Plot":
            num_for_box = st.selectbox("Select a numerical variable for box plot:", num_list, key="num_for_box")
            cat_for_box = st.selectbox("Select a categorical variable for grouping in box plot:", cat_list, key="cat_for_box")
            
            if num_for_box and cat_for_box:
                fig, ax = plt.subplots()
                sns.boxplot(x=cat_for_box, y=num_for_box, data=df, ax=ax)
                ax.set_title(f"Box Plot of {num_for_box} grouped by {cat_for_box}")
                st.pyplot(fig)
