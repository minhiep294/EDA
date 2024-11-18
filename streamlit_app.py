import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def univariate_analysis(df, num_list, cat_list):
    st.subheader("Univariate Analysis")
    variable_type = st.radio("Choose variable type:", ["Numerical", "Categorical"])
    if variable_type == "Numerical":
        col = st.selectbox("Select a numerical variable:", num_list)
        chart_type = st.selectbox("Choose chart type:", ["Histogram", "Box Plot", "Density Plot", "QQ Plot"])
        fig, ax = plt.subplots()
        if chart_type == "Histogram":
            sns.histplot(df[col], kde=True, ax=ax)
        elif chart_type == "Box Plot":
            sns.boxplot(x=df[col], ax=ax)
        elif chart_type == "Density Plot":
            sns.kdeplot(df[col], fill=True, ax=ax)
        elif chart_type == "QQ Plot":
            stats.probplot(df[col], dist="norm", plot=ax)
        ax.set_title(f"{chart_type} for {col}")
        st.pyplot(fig)
    elif variable_type == "Categorical":
        col = st.selectbox("Select a categorical variable:", cat_list)
        chart_type = st.selectbox("Choose chart type:", ["Count Plot", "Bar Chart", "Pie Chart"])
        fig, ax = plt.subplots()
        if chart_type == "Count Plot":
            sns.countplot(x=col, data=df, ax=ax)
        elif chart_type == "Bar Chart":
            df[col].value_counts().plot.bar(ax=ax)
        elif chart_type == "Pie Chart":
            df[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
        ax.set_title(f"{chart_type} for {col}")
        st.pyplot(fig)

def bivariate_analysis(df, num_list, cat_list):
    st.subheader("Bivariate Analysis")
    chart_type = st.selectbox("Choose chart type:", ["Scatter Plot", "Bar Plot", "Line Chart", "Correlation Coefficient"])
    if chart_type == "Scatter Plot":
        x = st.selectbox("Select Independent Variable (X, numerical):", num_list)
        y = st.selectbox("Select Dependent Variable (Y, numerical):", num_list)
        hue = st.selectbox("Optional Hue (categorical):", ["None"] + cat_list)
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue=None if hue == "None" else hue, data=df, ax=ax)
        ax.set_title(f"Scatter Plot: {y} vs {x}")
        st.pyplot(fig)
    elif chart_type == "Bar Plot":
        x = st.selectbox("Select Independent Variable (categorical):", cat_list)
        y = st.selectbox("Select Dependent Variable (numerical):", num_list)
        fig, ax = plt.subplots()
        sns.barplot(x=x, y=y, data=df, ax=ax)
        ax.set_title(f"Bar Plot: {y} grouped by {x}")
        st.pyplot(fig)
    elif chart_type == "Line Chart":
        x = st.selectbox("Select X-axis Variable (numerical or categorical):", df.columns)
        y = st.selectbox("Select Y-axis Variable (numerical):", num_list)
        fig, ax = plt.subplots()
        sns.lineplot(x=x, y=y, data=df, ax=ax)
        ax.set_title(f"Line Chart: {y} over {x}")
        st.pyplot(fig)
    elif chart_type == "Correlation Coefficient":
        x = st.selectbox("Select First Variable (numerical):", num_list)
        y = st.selectbox("Select Second Variable (numerical):", num_list)
        corr = df[x].corr(df[y])
        st.write(f"Correlation between {x} and {y}: {corr:.2f}")

def multivariate_analysis(df, num_list, cat_list):
    st.subheader("Multivariate Analysis")
    chart_type = st.selectbox("Choose chart type:", ["Pair Plot", "Correlation Matrix", "Grouped Bar Chart"])
    if chart_type == "Pair Plot":
        hue = st.selectbox("Optional Hue (categorical):", ["None"] + cat_list)
        sns.pairplot(df, hue=None if hue == "None" else hue)
        st.pyplot()
    elif chart_type == "Correlation Matrix":
        fig, ax = plt.subplots()
        sns.heatmap(df[num_list].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)
    elif chart_type == "Grouped Bar Chart":
        x = st.selectbox("Select X-axis Variable (categorical):", cat_list)
        hue = st.selectbox("Select Grouping Variable (categorical):", cat_list)
        fig, ax = plt.subplots()
        sns.countplot(x=x, hue=hue, data=df, ax=ax)
        ax.set_title(f"Grouped Bar Chart: {x} grouped by {hue}")
        st.pyplot(fig)

# App Layout
st.title("Interactive EDA Application")
uploaded_file = st.file_uploader("Upload your dataset (CSV only):")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    num_list = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_list = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio("Choose Analysis Type:", ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])


    if analysis_type == "Univariate Analysis":
        univariate_analysis(df, num_list, cat_list)
    elif analysis_type == "Bivariate Analysis":
        bivariate_analysis(df, num_list, cat_list)
    elif analysis_type == "Multivariate Analysis":
        multivariate_analysis(df, num_list, cat_list)
