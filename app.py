import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------- PAGE CONFIGURATION ----------------------------
st.set_page_config(
    page_title="TV Show Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        .main-title {text-align: center; font-size: 32px; font-weight: bold; color: #2E86C1;}
        .sub-title {text-align: center; font-size: 18px; color: #1C2833;}
        .stButton>button {background-color: #2E86C1; color: white; border-radius: 10px;}
        .footer {text-align: center; font-size: 14px; color: grey; margin-top: 30px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🎬 Enhancing User Experience through TV Show Recommendation & Visualization</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Developed by <b>Piyush Deepak Khodke</b> | DMW Mini Project 2025</p>', unsafe_allow_html=True)

# ---------------------------- LOAD DATA ----------------------------
@st.cache_data
def load_data():
    shows_df = pd.read_csv("tv_show_clustered_data.csv")
    rules_df = pd.read_csv("apriori_rules_tv_shows.csv")
    return shows_df, rules_df

shows_df, rules_df = load_data()

# ---------------------------- SIDEBAR NAVIGATION ----------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🎯 Recommendations", "📈 Clustering Insights", "📊 Dashboard", "ℹ️ About"])

# ====================================================================================
# 🏠 HOME PAGE
# ====================================================================================
if page == "🏠 Home":
    st.header("📺 Project Overview")
    st.write("""
    Welcome to the **TV Show Recommendation System** web app!  
    This project enhances user experience by combining **Apriori Association Rule Mining** 
    and **K-Means Clustering** for user segmentation and content recommendations.
    
    ### 🧩 Features:
    - Personalized show suggestions based on your watch history  
    - Cluster prediction using your viewing behavior  
    - Interactive data visualizations  
    - Exportable recommendations  
    """)

    # Show a quick overview chart
    st.subheader("🎞️ Top 5 Genres by Average Satisfaction")
    genre_avg = shows_df.groupby('Genre')['User_Satisfaction_Score'].mean().sort_values(ascending=False).head(5)
    st.bar_chart(genre_avg)

    st.markdown("---")
    st.metric("Total Users", len(shows_df['User_ID'].unique()))
    st.metric("Total Shows", len(shows_df['TV_Show_Name'].unique()))

# ====================================================================================
# 🎯 RECOMMENDATION SYSTEM
# ====================================================================================
elif page == "🎯 Recommendations":
    st.header("🎯 Personalized TV Show Recommendations")

    # Sidebar inputs
    st.sidebar.subheader("User Preferences")
    genre_list = sorted(shows_df['Genre'].dropna().unique())
    selected_genre = st.sidebar.selectbox("Select your favorite genre:", genre_list)
    watched_shows = st.sidebar.multiselect("Select shows you’ve already watched:", sorted(shows_df['TV_Show_Name'].dropna().unique()))

    # Genre-based suggestions
    st.subheader("🎬 Top Shows in Your Favorite Genre")
    genre_shows = shows_df[shows_df['Genre'] == selected_genre]['TV_Show_Name'].unique()
    st.write(", ".join(list(genre_shows[:10])) if len(genre_shows) > 0 else "No shows found for this genre.")

    # Apriori-based personalized recommendations
    st.subheader("🎥 Recommended Shows Based on Watched History")
    recommendations = []

    if watched_shows:
        for show in watched_shows:
            matches = rules_df[rules_df['antecedents'].apply(lambda x: show in x)]
            if not matches.empty:
                recs = matches.sort_values(by='lift', ascending=False)['consequents'].head(3)
                for r in recs:
                    for item in eval(r) if isinstance(r, str) else r:
                        recommendations.append(item)

        recommendations = list(set(recommendations) - set(watched_shows))

        if recommendations:
            st.success("✨ Based on your watching history, you might also like:")
            for rec in recommendations[:5]:
                st.markdown(f"- 🎬 **{rec}**")

            # Downloadable recommendations
            rec_df = pd.DataFrame({"Recommended Shows": recommendations})
            st.download_button("⬇️ Download Recommendations", rec_df.to_csv(index=False), "recommendations.csv", "text/csv")
        else:
            st.warning("No strong recommendations found for your selected shows.")
    else:
        st.info("Select your watched shows in the sidebar to get personalized recommendations.")

# ====================================================================================
# 📈 CLUSTERING INSIGHTS
# ====================================================================================
elif page == "📈 Clustering Insights":
    st.header("📈 K-Means Clustering Analysis")

    features = ['Watch_Time_Hours', 'User_Satisfaction_Score']
    X = shows_df[features].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42)
    shows_df['Cluster'] = kmeans.fit_predict(X)

    # Cluster summary
    st.subheader("📊 Cluster Summary")
    cluster_summary = shows_df.groupby('Cluster')[['Watch_Time_Hours', 'User_Satisfaction_Score']].mean().round(2)
    st.dataframe(cluster_summary)

    # Visualization
    st.subheader("🎨 Cluster Visualization")
    fig = px.scatter(
        shows_df,
        x='Watch_Time_Hours', y='User_Satisfaction_Score',
        color='Cluster',
        hover_data=['TV_Show_Name', 'Genre', 'Platform'],
        title="User Clusters by Viewing Behavior",
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------ USER INPUT SECTION ------------------------------------
    st.markdown("---")
    st.subheader("🧩 Predict Your Cluster Type")

    watch_time = st.number_input("Enter your average watch duration (in hours):", min_value=0.1, max_value=1.0, value=0.5)
    satisfaction = st.slider("Rate your satisfaction (0–1):", 0.0, 1.0, 0.8)
    platform = st.selectbox("Preferred Platform:", sorted(shows_df['Platform'].dropna().unique()))
    genre = st.selectbox("Favorite Genre:", sorted(shows_df['Genre'].dropna().unique()))

    user_data = pd.DataFrame([[watch_time, satisfaction]], columns=['Watch_Time_Hours', 'User_Satisfaction_Score'])
    user_cluster = kmeans.predict(user_data)[0]

    st.success(f"🎯 You belong to **Cluster {user_cluster}** based on your viewing behavior!")

    # Cluster interpretation
    if user_cluster == 0:
        st.info("📺 **Cluster 0:** High-duration, high-satisfaction users — loyal and binge-watchers.")
    elif user_cluster == 1:
        st.info("🎞️ **Cluster 1:** Moderate watchers — casual entertainment consumers.")
    else:
        st.info("🎬 **Cluster 2:** Low-duration, low-satisfaction users — new or less engaged viewers.")

    # Visualize user's position
    st.subheader("📊 Your Position on Cluster Graph")
    fig2, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=shows_df, x='Watch_Time_Hours', y='User_Satisfaction_Score', hue='Cluster', palette='viridis', s=100)
    plt.scatter(watch_time, satisfaction, color='red', s=200, edgecolor='black', label='You')
    plt.legend()
    plt.title("User Clusters and Your Position")
    st.pyplot(fig2)


# ====================================================================================
# 📊 DASHBOARD PAGE (With OLAP Slicers)
# ====================================================================================
elif page == "📊 Dashboard":
    st.header("📊 Data Insights Dashboard")

    st.write("""
    Interactive visualizations of viewing patterns, genre preferences, and user satisfaction levels.
    Use the **OLAP-style slicer** below to filter data by platform.
    """)

    # --------------------- OLAP SLICER ---------------------
    platforms = sorted(shows_df['Platform'].dropna().unique())
    selected_platform = st.multiselect("🎛️ Select Platform(s) to Filter:", platforms, default=platforms)

    # Filter data based on selected platform(s)
    filtered_df = shows_df[shows_df['Platform'].isin(selected_platform)]

    # If no platform selected
    if filtered_df.empty:
        st.warning("⚠️ Please select at least one platform to view dashboard data.")
    else:
        # ------------------ DASHBOARD VISUALS ------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 Genre-wise Average Satisfaction")
            chart1 = px.bar(
                filtered_df.groupby('Genre')['User_Satisfaction_Score'].mean().reset_index(),
                x='Genre', y='User_Satisfaction_Score',
                color='Genre', title=f"Average Satisfaction per Genre ({', '.join(selected_platform)})"
            )
            st.plotly_chart(chart1, use_container_width=True)

        with col2:
            st.subheader("📺 Platform-wise Average Watch Duration")
            chart2 = px.bar(
                filtered_df.groupby('Platform')['Watch_Time_Hours'].mean().reset_index(),
                x='Platform', y='Watch_Time_Hours',
                color='Platform', title="Average Watch Duration per Platform"
            )
            st.plotly_chart(chart2, use_container_width=True)

        # Scatter plot
        st.subheader("🧭 Satisfaction vs Duration by Genre")
        chart3 = px.scatter(
            filtered_df, x='Watch_Time_Hours', y='User_Satisfaction_Score',
            color='Genre', hover_data=['TV_Show_Name', 'Platform'],
            title=f"Satisfaction vs Duration ({', '.join(selected_platform)})"
        )
        st.plotly_chart(chart3, use_container_width=True)

        # ------------------ METRICS ------------------
        st.markdown("---")
        st.subheader("📊 Summary Metrics")

        total_users = filtered_df['User_ID'].nunique()
        total_shows = filtered_df['TV_Show_Name'].nunique()
        avg_watch = round(filtered_df['Watch_Time_Hours'].mean(), 2)
        avg_satisfaction = round(filtered_df['User_Satisfaction_Score'].mean(), 2)

        colm1, colm2, colm3, colm4 = st.columns(4)
        colm1.metric("👥 Unique Users", total_users)
        colm2.metric("🎞️ Total Shows", total_shows)
        colm3.metric("⏱️ Avg Watch Time (hrs)", avg_watch)
        colm4.metric("⭐ Avg Satisfaction", avg_satisfaction)


# ====================================================================================
# ℹ️ ABOUT PAGE
# ====================================================================================
elif page == "ℹ️ About":
    st.header("ℹ️ About the Developer")
    st.write("""
    **Developer:** Piyush Deepak Khodke  
    **Project:** DMW Mini Project 2025  
    **Objective:** Enhancing User Experience through TV Show Recommendation and Visualization  
    **Tech Stack:** Python, Streamlit, Apriori Algorithm, K-Means, Plotly, Seaborn  

    This web app demonstrates how data mining and clustering techniques can be used 
    to recommend TV shows, segment users, and visualize audience behavior effectively.
    """)
    st.markdown('<p class="footer">© 2025 Piyush Deepak Khodke | Designed for Mini Project Submission</p>', unsafe_allow_html=True)
