import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import time



# Customizing the page layout and theme
st.set_page_config(page_title="Data Balancer", layout="wide", page_icon="🔄")

st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        color: #333;
    }
    .stApp {
        max-width: 100%;  /* Increase width */
        margin: auto;
    }
    .css-18e3th9 {
        padding: 2rem 8rem; /* Increase left and right padding */
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        border-radius: 8px;
    }
    .stSelectbox, .stRadio, .stDataFrame, .stTextInput {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# Title of the app
st.markdown(
    """
    <h1 style="font-size:35px; font-weight: bold; display: flex; align-items: center;">
        🔄 Synthetic Data Generator: Fix Imbalanced Data with SMOTE
    </h1>
    """,
    unsafe_allow_html=True
)

# Sample dataset
sample_data_path = "sample_imbalanced_data_missing_data.csv"

def load_sample_data():
    return pd.read_csv(sample_data_path)

# Option to choose sample data or upload new data
data_option = st.radio("Choose dataset option:", ["Use Sample Data", "Upload New Data"], horizontal=True)

if data_option == "Upload New Data":
    uploaded_file = st.file_uploader("📂 Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]

        # Read dataset based on extension
        with st.spinner("⏳ Loading dataset..."):
            time.sleep(1)
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file)

        # Validate dataset format
        if df.select_dtypes(include=['object']).shape[1] > 0:
            st.error("❌ The uploaded dataset contains non-numeric values. Please upload a dataset with numerical features only.")
            df = None
else:
    with st.spinner("⏳ Loading sample dataset..."):
        time.sleep(1)
        df = load_sample_data()
        st.success("✅ Using sample dataset.")

if 'df' in locals() and df is not None:
    with st.spinner("🔍 Generating dataset preview..."):
        time.sleep(1)
        st.subheader("📋 Dataset Preview:")
        st.dataframe(df.head())

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Missing values detected in the dataset.")
        st.write("### Missing Values Summary")
        st.write(df.isnull().sum().rename('Missing Value Counts'))
        missing_option = st.selectbox("🧹 Choose a missing value handling strategy:", ["None", "Drop Rows", "Fill with Mean", "Fill with Median","Fill with Mode"], key='missing_strategy')
        if missing_option == "Drop Rows":
            df.dropna(inplace=True)
            st.success("✅ Missing values have been removed.")
        elif missing_option == "Fill with Mean":
            df.fillna(df.mean(), inplace=True)
            st.success("✅ Missing values have been filled with column means.")
        elif missing_option == "Fill with Median":
            df.fillna(df.median(), inplace=True)
            st.success("✅ Missing values have been filled with column medians.")
        elif missing_option == "Fill with Mode":
            df.fillna(df.mode().iloc[0], inplace=True)
            st.success("✅ Missing values have been filled with column modes.")



    # Step 2: Select target column
    target_column = st.selectbox("🎯 Select the target variable:", [None] + list(df.columns), index=0, key='target_variable')

    if target_column is not None:
        with st.spinner("📊 Generating value counts..."):
            time.sleep(1)
            st.subheader("📈 Target Variable Distribution")
            st.write(df[target_column].value_counts())


        # Step 3: Select balance ratio
        balance_ratio = st.radio("⚖ Choose balance ratio:", [None, "50:50", "70:30", "80:20"], horizontal=True, index=0)

        if balance_ratio:
            ratio_map = {
                "50:50": 1.0,
                "80:20": 0.8,
                "70:30": 0.7
            }
            sampling_strategy = ratio_map[balance_ratio]

            # Apply SMOTE balancing
            with st.spinner("⚙ Applying SMOTE balancing..."):
                time.sleep(1)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)

                df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_column])], axis=1)

            # Prepare data for pie chart hover info
            df_pie_before = df[target_column].value_counts().reset_index()
            df_pie_before.columns = [target_column, 'count']
            df_pie_before['percentage'] = (df_pie_before['count'] / df_pie_before['count'].sum()) * 100
            df_pie_before['info'] = df_pie_before.apply(lambda row: f"Target Variable: {row[target_column]}<br>Count: {row['count']}<br>Percentage: {row['percentage']:.2f}%", axis=1)

            df_pie_after = df_balanced[target_column].value_counts().reset_index()
            df_pie_after.columns = [target_column, 'count']
            df_pie_after['percentage'] = (df_pie_after['count'] / df_pie_after['count'].sum()) * 100
            df_pie_after['info'] = df_pie_after.apply(lambda row: f"Target Variable: {row[target_column]}<br>Count: {row['count']}<br>Percentage: {row['percentage']:.2f}%", axis=1)

            # Visualizations
            st.subheader("📊 Class Distribution Before and After Balancing")

            chart_type = st.radio("Select chart type:", ["Pie Chart", "Bar Chart"], horizontal=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3 style='font-size:16px;'><b>Before Balancing</b></h3>", unsafe_allow_html=True)
                if chart_type == "Pie Chart":
                    fig = px.pie(df_pie_before, names=target_column, values='count',
                         hover_data={'info': True}, custom_data=['info'])
                    fig.update_traces(hovertemplate='%{customdata[0]}')
                    st.plotly_chart(fig)

                else:
                    df_bar_before = df[target_column].value_counts().reset_index()
                    df_bar_before.columns = ['Class', 'Count']
                    df_bar_before['Percentage'] = (df_bar_before['Count'] / df_bar_before['Count'].sum()) * 100
                    fig = px.bar(df_bar_before, x='Class', y='Count', text=df_bar_before['Percentage'].apply(lambda x: f'{x:.2f}%'),
                        color='Class', color_discrete_sequence=px.colors.qualitative.Set1)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig)

            with col2:
                st.markdown("<h3 style='font-size:16px;'><b>After Balancing</b></h3>", unsafe_allow_html=True)
                if chart_type == "Pie Chart":
                    fig = px.pie(df_pie_after, names=target_column, values='count',
                                 hover_data={'info': True}, custom_data=['info'])
                    fig.update_traces(hovertemplate='%{customdata[0]}')
                    st.plotly_chart(fig)
                else:
                    df_bar_after = df_balanced[target_column].value_counts().reset_index()
                    df_bar_after.columns = ['Class', 'Count']
                    df_bar_after['Percentage'] = (df_bar_after['Count'] / df_bar_after['Count'].sum()) * 100
                    fig = px.bar(df_bar_after, x='Class', y='Count', text=df_bar_after['Percentage'].apply(lambda x: f'{x:.2f}%'),
                        color='Class', color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig)


            st.subheader("📊 Heatmap Correlation Analysis")
            col3, col4 = st.columns(2)
            with col3:
                #st.write("### Heatmap Before Balancing")
                st.markdown("<h3 style='font-size:16px;'><b>Before Balancing</b></h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), annot=True, cmap='Blues', ax=ax)
                st.pyplot(fig)
            with col4:
                #st.write("### Heatmap After Balancing")
                st.markdown("<h3 style='font-size:16px;'><b>After Balancing</b></h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                sns.heatmap(df_balanced.corr(), annot=True, cmap='Blues', ax=ax)
                st.pyplot(fig)

            st.markdown("<br><br>", unsafe_allow_html=True)

            
            st.subheader("📊 Summary Description")
            # Layout with centered columns for data summary
            col7, col8 = st.columns([1, 1])
            with col7:
                st.markdown("<h3 style='font-size:16px;'><b>Before Balancing</b></h3>", unsafe_allow_html=True)
                st.write(df.describe())
            with col8:
                st.markdown("<h3 style='font-size:16px;'><b>After Balancing</b></h3>", unsafe_allow_html=True)
                st.write(df_balanced.describe())


            # Center align the download button
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.write("### ⬇ Download Processed Dataset")
            st.markdown(
                '<a href="data:file/csv;base64,' + df_balanced.to_csv(index=False).encode().hex() + '" download="balanced_data.csv">'
                '<button style="background-color:#4CAF50; color:white; padding:10px 24px; border:none; border-radius:8px; cursor:pointer;">Download Balanced CSV</button>'
                '</a>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)


st.sidebar.subheader("ℹ️ About this App")
st.sidebar.info("""
👋 **Welcome!**  \n\n\n
Effortlessly preprocess and balance your datasets with **SMOTE** for improved model performance.
\n
**Why Synthetic Data?**  
- Bias Reduction  
- Privacy Protection  
- Increased Data Volume  
- Diverse Scenario Generation  
- Cost-Effective  
\n
**Quick Start:**  
1. Upload your dataset or use the sample.  
2. Handle missing values.  
3. Select the target variable for balancing.  
4. Compare class distributions.  
5. Download the processed dataset.  

\n\n \n \n \n\n\n
📧 **Contact:** kshethia11@gmail.com  
""")
