import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize the app
st.title("EDA with Streamlit")
st.write("Upload a dataset to explore its variables with various charts.")

# File upload section
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Identify numerical and categorical columns
    num_list = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_list = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    st.write("Numerical Columns:", num_list)
    st.write("Categorical Columns:", cat_list)

    # Sidebar menu for navigation
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Data Cleaning & Descriptive Stats", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"]
    )

    # Data Cleaning & Descriptive Stats
    if analysis_type == "Data Cleaning & Descriptive Stats":
        st.header("1. Data Cleaning")
        st.subheader("Handle Missing Values")
        missing_option = st.radio(
            "Choose a method to handle missing values:",
            ("Impute with Mean", "Remove Rows with Missing Data", "Leave as is")
        )
        if missing_option == "Impute with Mean":
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
            col_type = st.selectbox(
                f"Select data type for {col}",
                ("Automatic", "Integer", "Float", "String", "DateTime"), index=0
            )
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

        st.header("2. Descriptive Statistics")
        st.subheader("Central Tendency & Dispersion")
        st.write(df.describe(include='all'))

        if st.checkbox("Show Mode"):
            st.write(df.mode().iloc[0])

    # Univariate Analysis
    elif analysis_type == "Univariate Analysis":
        st.header("Univariate Analysis")

        st.subheader("Numerical Data Visualization")
        num_col = st.selectbox("Select a numerical variable:", num_list)
        if num_col:
            chart_type = st.selectbox(
                "Select chart type:",
                ["Histogram", "Box Plot", "Density Plot", "Area Plot", "Dot Plot", "Frequency Polygon", "QQ Plot"]
            )

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
            elif chart_type == "Area Plot":
                sns.histplot(df[num_col], kde=True, fill=True, element="poly", ax=ax)
                ax.set_title(f"Area Plot of {num_col}")
            elif chart_type == "Dot Plot":
                sns.stripplot(x=df[num_col], jitter=True, ax=ax)
                ax.set_title(f"Dot Plot of {num_col}")
            elif chart_type == "Frequency Polygon":
                sns.histplot(df[num_col], kde=False, element="step", color="blue", ax=ax)
                ax.set_title(f"Frequency Polygon of {num_col}")
            elif chart_type == "QQ Plot":
                from scipy import stats
                stats.probplot(df[num_col], dist="norm", plot=ax)
                ax.set_title(f"QQ Plot of {num_col}")
            st.pyplot(fig)

        st.subheader("Categorical Data Visualization")
        cat_col = st.selectbox("Select a categorical variable:", cat_list)
        if cat_col:
            cat_chart_type = st.selectbox(
                "Select chart type:",
                ["Count Plot", "Bar Chart", "Pie Plot"]
            )

            fig, ax = plt.subplots()
            if cat_chart_type == "Count Plot":
                sns.countplot(x=df[cat_col], ax=ax)
                ax.set_title(f"Count Plot of {cat_col}")
            elif cat_chart_type == "Bar Chart":
                sns.barplot(
                    x=df[cat_col].value_counts().index,
                    y=df[cat_col].value_counts().values,
                    ax=ax
                )
                ax.set_title(f"Bar Chart of {cat_col}")
            elif cat_chart_type == "Pie Plot":
                df[cat_col].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_ylabel('')
                ax.set_title(f"Pie Plot of {cat_col}")
            st.pyplot(fig)

    # Bivariate Analysis
    elif analysis_type == "Bivariate Analysis":
        st.header("Bivariate Analysis")
        # Guidance for users
    st.write("""
    **Guidance:**
    - Pair Plot: For numerical variables only.
    - Scatter Plot: Choose two numerical variables. Optional: Color by a categorical variable, size by another numerical variable.
    - Correlation Coefficient: Choose two numerical variables to compute correlation.
    - Bar Plot: Select one categorical variable and one numerical variable.
    - Line Chart: Select one X-axis variable (commonly time or index) and one or two numerical variables for Y-axis.
    - Stacked Bar Chart: Select two categorical variables.
    """)

    # Choose Chart Type
    bivar_chart_type = st.selectbox(
        "Select chart type:",
        ["Pair Plot", "Scatter Plot", "Correlation Coefficient", "Bar Plot", "Line Chart", "Stacked Bar Chart"]
    )

    # Pair Plot
    if bivar_chart_type == "Pair Plot":
        st.write("### Pair Plot (Numerical Variables Only)")
        if len(num_list) > 1:
            st.write("Select this to see pairwise scatter plots for all numerical variables.")
            pair_plot = sns.pairplot(df[num_list])
            st.pyplot(pair_plot.fig)
        else:
            st.warning("Not enough numerical variables for a pair plot.")

    # Scatter Plot
    elif bivar_chart_type == "Scatter Plot":
        st.write("### Scatter Plot (Two Numerical Variables Required)")
        scatter_x = st.selectbox("Select X-axis variable (numerical):", num_list, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis variable (numerical):", num_list, key="scatter_y")
        scatter_color = st.selectbox("Select a categorical variable for color (optional):", [""] + cat_list, key="scatter_color")
        scatter_size = st.selectbox("Select numerical variable for size (optional):", [""] + num_list, key="scatter_size")
        
        if scatter_x and scatter_y:
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=df[scatter_x],
                y=df[scatter_y],
                hue=df[scatter_color] if scatter_color else None,
                size=df[scatter_size] if scatter_size else None,
                data=df,
                ax=ax
            )
            ax.set_title(f"Scatter Plot: {scatter_x} vs {scatter_y}")
            st.pyplot(fig)

    # Correlation Coefficient
    elif bivar_chart_type == "Correlation Coefficient":
        st.write("### Correlation Coefficient (Two Numerical Variables Required)")
        num_x_corr = st.selectbox("Select first numerical variable:", num_list, key="num_x_corr")
        num_y_corr = st.selectbox("Select second numerical variable:", num_list, key="num_y_corr")
        
        if num_x_corr and num_y_corr:
            corr_value = df[num_x_corr].corr(df[num_y_corr])
            st.write(f"**Correlation between {num_x_corr} and {num_y_corr}:** {corr_value:.2f}")

    # Bar Plot
    elif bivar_chart_type == "Bar Plot":
        st.write("### Bar Plot (One Categorical and One Numerical Variable Required)")
        cat_for_bar = st.selectbox("Select a categorical variable for grouping:", cat_list, key="cat_for_bar")
        num_for_bar = st.selectbox("Select a numerical variable for values:", num_list, key="num_for_bar")
        
        if cat_for_bar and num_for_bar:
            fig, ax = plt.subplots()
            sns.barplot(x=cat_for_bar, y=num_for_bar, data=df, ax=ax)
            ax.set_title(f"Bar Plot: {num_for_bar} by {cat_for_bar}")
            st.pyplot(fig)

    # Line Chart
    elif bivar_chart_type == "Line Chart":
        st.write("### Line Chart (One X-axis Variable and One/Two Numerical Variables Required)")
        line_x = st.selectbox("Select variable for X-axis (commonly time or index):", [""] + list(df.columns), key="line_x")
        line_y1 = st.selectbox("Select first numerical variable for Y-axis:", num_list, key="line_y1")
        line_y2 = st.selectbox("Select second numerical variable for Y-axis (optional):", [""] + num_list, key="line_y2")
        
        if line_x and line_y1:
            fig, ax = plt.subplots()
            sns.lineplot(x=df[line_x], y=df[line_y1], ax=ax, label=line_y1)
            if line_y2:
                sns.lineplot(x=df[line_x], y=df[line_y2], ax=ax, label=line_y2)
            ax.set_title(f"Line Chart: {line_y1}" + (f" and {line_y2}" if line_y2 else "") + f" over {line_x}")
            st.pyplot(fig)

    # Stacked Bar Chart
    elif bivar_chart_type == "Stacked Bar Chart":
        st.write("### Stacked Bar Chart (Two Categorical Variables Required)")
        stack_cat1 = st.selectbox("Select first categorical variable for grouping:", cat_list, key="stack_cat1")
        stack_cat2 = st.selectbox("Select second categorical variable for stacking:", cat_list, key="stack_cat2")
        
        if stack_cat1 and stack_cat2:
            stacked_data = df.groupby([stack_cat1, stack_cat2]).size().unstack(fill_value=0)
            fig, ax = plt.subplots()
            stacked_data.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"Stacked Bar Chart: {stack_cat1} by {stack_cat2}")
            st.pyplot(fig)

    # Multivariate Analysis
    elif analysis_type == "Multivariate Analysis":
        st.header("Multivariate Analysis")
        # Guidance for users
    st.write("""
    **Guidance:**
    - Correlation Matrix: For numerical variables only.
    - Pair Plot: For numerical variables only.
    - Grouped Bar Chart: For two categorical variables.
    - Pair Plot with Hue: For numerical variables with an additional categorical variable for grouping (hue).
    - Box Plot: For one numerical variable and one categorical variable for grouping.
    """)

    # Select Chart Type
    multi_chart_type = st.selectbox(
        "Select chart type:",
        ["Correlation Matrix", "Pair Plot", "Grouped Bar Chart", "Pair Plot with Hue", "Box Plot"]
    )

    # Correlation Matrix
    if multi_chart_type == "Correlation Matrix":
        st.write("### Correlation Matrix (Numerical Variables Only)")
        if len(num_list) > 1:
            corr_matrix = df[num_list].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix for Numerical Variables")
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical variables for a correlation matrix.")

    # Pair Plot
    elif multi_chart_type == "Pair Plot":
        st.write("### Pair Plot (Numerical Variables Only)")
        if len(num_list) > 1:
            st.write("This chart visualizes pairwise scatter plots for all numerical variables.")
            pair_plot = sns.pairplot(df[num_list])
            st.pyplot(pair_plot.fig)
        else:
            st.warning("Not enough numerical variables for a pair plot.")

    # Grouped Bar Chart
    elif multi_chart_type == "Grouped Bar Chart":
        st.write("### Grouped Bar Chart (Two Categorical Variables Required)")
        cat_x = st.selectbox("Select X-axis categorical variable:", cat_list, key="cat_x_grouped")
        cat_hue = st.selectbox("Select a categorical variable for grouping (hue):", cat_list, key="cat_hue_grouped")
        
        if cat_x and cat_hue:
            fig, ax = plt.subplots()
            sns.countplot(x=cat_x, hue=cat_hue, data=df, ax=ax)
            ax.set_title(f"Grouped Bar Chart: {cat_x} grouped by {cat_hue}")
            st.pyplot(fig)
        else:
            st.warning("Please select valid categorical variables for the grouped bar chart.")

    # Pair Plot with Hue
    elif multi_chart_type == "Pair Plot with Hue":
        st.write("### Pair Plot with Hue (Numerical Variables and One Categorical Variable)")
        hue_cat = st.selectbox("Select a categorical variable for hue:", cat_list, key="hue_cat_for_pairplot")
        if hue_cat:
            st.write("This chart shows pairwise scatter plots for numerical variables, grouped by the selected categorical variable.")
            pair_plot_hue = sns.pairplot(df, vars=num_list, hue=hue_cat)
            st.pyplot(pair_plot_hue.fig)
        else:
            st.warning("Please select a valid categorical variable for hue.")

    # Box Plot
    elif multi_chart_type == "Box Plot":
        st.write("### Box Plot (Numerical Variable Grouped by a Categorical Variable)")
        num_for_box = st.selectbox("Select a numerical variable:", num_list, key="num_for_box")
        cat_for_box = st.selectbox("Select a categorical variable for grouping:", cat_list, key="cat_for_box")
        
        if num_for_box and cat_for_box:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat_for_box, y=num_for_box, data=df, ax=ax)
            ax.set_title(f"Box Plot: {num_for_box} grouped by {cat_for_box}")
            st.pyplot(fig)
        else:
            st.warning("Please select valid variables for the box plot.")
