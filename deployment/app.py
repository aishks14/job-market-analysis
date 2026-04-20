"""
Job Market Analysis — Streamlit Web Application
================================================
Run with:  streamlit run deployment/app.py

Features:
  - Job recommendation engine (TF-IDF + cosine similarity)
  - Market overview dashboard
  - Category & country analytics
  - Future trend predictions
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

# ── Try importing Streamlit ──────────────────────────────────
try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Upwork Job Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d1117, #1a2332);
        padding: 2rem; border-radius: 10px; margin-bottom: 1.5rem;
        border: 1px solid #30363d;
    }
    .main-header h1 {
        color: #58a6ff; font-size: 2rem; margin-bottom: 0.3rem;
    }
    .main-header p { color: #8b949e; font-size: 1rem; }
    .metric-card {
        background: #161b22; padding: 1.2rem; border-radius: 8px;
        border: 1px solid #30363d; text-align: center;
    }
    .metric-card .metric-val {
        font-size: 1.8rem; font-weight: 700; color: #58a6ff;
    }
    .metric-card .metric-label { color: #8b949e; font-size: 0.85rem; }
    .rec-card {
        background: #161b22; padding: 0.9rem 1.2rem; border-radius: 8px;
        border: 1px solid #30363d; margin-bottom: 0.5rem;
    }
    .rec-card .rec-title { color: #e6edf3; font-weight: 600; }
    .rec-card .rec-meta  { color: #8b949e; font-size: 0.82rem; }
    .rec-card .rec-score { color: #3fb950; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'upwork_clean.csv')
VEC_PATH  = os.path.join(ROOT_DIR, 'models', 'tfidf_vectorizer.pkl')
MAT_PATH  = os.path.join(ROOT_DIR, 'models', 'tfidf_matrix.npz')

# ── Load data & models ───────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['published_date'] = pd.to_datetime(df['published_date'], utc=True)
    return df

@st.cache_resource
def load_models():
    vec = pickle.load(open(VEC_PATH, 'rb'))
    mat = sp.load_npz(MAT_PATH)
    return vec, mat

# ── Check if processed data exists ───────────────────────────
if not os.path.exists(DATA_PATH):
    st.error("⚠️ Processed data not found. Please run notebooks in order first (01 → 05).")
    st.stop()

df = load_data()

# Load models if available
models_ready = os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH)
if models_ready:
    vec, mat = load_models()
    df_rec = df.sort_values('published_date', ascending=False).head(60000).copy().reset_index(drop=True)
    df_rec['text_for_index'] = df_rec['title'].fillna('') + ' ' + df_rec['category'].fillna('')

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "🔍 Job Recommender",
    "💰 Salary Analysis",
    "🌍 Country Insights",
    "📈 Trends & Forecast"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Stats**")
st.sidebar.markdown(f"- Total jobs: **{len(df):,}**")
st.sidebar.markdown(f"- Countries: **{df['country'].nunique()}**")
st.sidebar.markdown(f"- Categories: **{df['category'].nunique()}**")
st.sidebar.markdown(f"- Date range: Nov 2023 – Mar 2024")

# ═══════════════════════ PAGE: OVERVIEW ════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
      <h1>Upwork Job Market Analyzer</h1>
      <p>Comprehensive analysis of 244,828 job postings from Nov 2023 – Mar 2024 across 212 countries</p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, f"{len(df):,}", "Total Jobs"),
        (col2, f"{df['country'].nunique()}", "Countries"),
        (col3, f"{df['category'].nunique()}", "Categories"),
        (col4, f"${df[df['is_hourly'] & df['avg_hourly'].between(3,200)]['avg_hourly'].median():.0f}/hr", "Median Rate"),
        (col5, f"{df['is_hourly'].mean()*100:.0f}%", "Hourly Jobs"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Job Type Distribution")
        type_counts = df['job_type'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        colors = ['#4C72B0', '#DD8452']
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90,
               wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2},
               textprops={'color': '#e6edf3', 'fontsize': 11})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.subheader("Monthly Posting Volume")
        monthly = df.groupby('year_month').size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        ax.bar(monthly['year_month'], monthly['count'], color='#4C72B0', edgecolor='#0d1117')
        ax.set_xlabel('Month', color='#8b949e')
        ax.set_ylabel('Postings', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
        ax.spines[['top','right']].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_color('#30363d')
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Top Job Categories")
    cat_counts = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    colors = sns.color_palette('tab10', n_colors=len(cat_counts))
    ax.barh(cat_counts.index[::-1], cat_counts.values[::-1],
            color=colors[::-1], edgecolor='#0d1117', height=0.7)
    ax.set_xlabel('Number of Postings', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax.spines[['top','right']].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════ PAGE: RECOMMENDER ═════════════════════
elif page == "🔍 Job Recommender":
    st.title("🔍 Job Recommendation Engine")
    st.markdown("*Enter your skills and get personalized Upwork job matches using TF-IDF + Cosine Similarity*")

    if not models_ready:
        st.warning("Recommendation models not found. Run Notebook 05 first to generate them.")
        st.stop()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Your skills / desired role", 
                              placeholder="e.g. Python machine learning data scientist")
    with col2:
        job_type_f = st.selectbox("Job Type", ["All", "Hourly", "Fixed-Price"])
    with col3:
        top_n = st.slider("Results", 5, 20, 10)

    country_f = st.text_input("Filter by country (optional)", 
                               placeholder="e.g. United States")

    if st.button("Find Jobs", type="primary") and query:
        query_vec = vec.transform([query])
        scores    = cosine_similarity(query_vec, mat).ravel()
        result    = df_rec.copy()
        result['similarity'] = scores

        if job_type_f != "All":
            result = result[result['job_type'] == job_type_f]
        if country_f.strip():
            result = result[result['country'].str.lower() == country_f.strip().lower()]

        top = result.nlargest(top_n, 'similarity').reset_index(drop=True)
        st.success(f"Found {len(top)} matches for: **{query}**")

        for _, row in top.iterrows():
            rate_str = (f"${row['avg_hourly']:.0f}/hr" if pd.notna(row.get('avg_hourly'))
                        else f"${row['budget']:.0f} fixed" if pd.notna(row.get('budget'))
                        else "Rate not specified")
            st.markdown(f"""
            <div class="rec-card">
              <div class="rec-title">{row['title']}</div>
              <div class="rec-meta">
                {row['category']}  &nbsp;|&nbsp;
                {row['job_type']}  &nbsp;|&nbsp;
                {rate_str}  &nbsp;|&nbsp;
                {row['country']}
              </div>
              <div class="rec-score">Similarity: {row['similarity']:.4f}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════ PAGE: SALARY ══════════════════════════
elif page == "💰 Salary Analysis":
    st.title("Salary & Budget Analysis")
    tab1, tab2 = st.tabs(["Hourly Rate by Category", "Fixed-Price Budget"])

    with tab1:
        hourly_sub = df[df['is_hourly'] & df['avg_hourly'].between(3, 200)].copy()
        top_cats   = hourly_sub['category'].value_counts().head(10).index
        order      = (hourly_sub[hourly_sub['category'].isin(top_cats)]
                      .groupby('category')['avg_hourly']
                      .median().sort_values(ascending=False).index)
        data_plot  = hourly_sub[hourly_sub['category'].isin(top_cats)].copy()
        data_plot['category'] = pd.Categorical(data_plot['category'], categories=order, ordered=True)

        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        palette = sns.color_palette('RdYlGn', n_colors=len(order))
        sns.boxplot(data=data_plot.sort_values('category'), x='avg_hourly', y='category',
                    palette=palette, width=0.55, fliersize=1.5, ax=ax)
        ax.set_xlabel('Average Hourly Rate (USD)', color='#8b949e')
        ax.set_ylabel('Category', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        ax.spines[['top','right']].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fixed_sub = df[~df['is_hourly'] & df['budget_capped'].notna()].copy()
        order_f   = (fixed_sub.groupby('category')['budget']
                     .median().sort_values(ascending=False).head(8).index)
        fixed_top = fixed_sub[fixed_sub['category'].isin(order_f)]

        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        sns.violinplot(data=fixed_top, y='category', x='budget_capped',
                       order=order_f, palette='muted', inner='quartile',
                       orient='h', ax=ax)
        ax.set_xlabel('Budget (USD, capped)', color='#8b949e')
        ax.set_ylabel('Category', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        ax.spines[['top','right']].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════ PAGE: COUNTRY ═════════════════════════
elif page == "🌍 Country Insights":
    st.title("Country Rate Insights")
    country_data = (
        df[df['is_hourly'] & df['avg_hourly'].between(3, 200) & (df['country'] != 'Unknown')]
        .groupby('country')['avg_hourly']
        .agg(median='median', count='count')
        .query('count >= 20')
        .sort_values('median', ascending=False)
    )
    n_show = st.slider("Number of countries to display", 10, 30, 20)
    top_c  = country_data.head(n_show)
    global_med = country_data['median'].median()

    fig, ax = plt.subplots(figsize=(12, max(5, n_show * 0.35)), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    palette = sns.color_palette('RdYlGn', n_colors=len(top_c))
    bars = ax.barh(top_c.index[::-1], top_c['median'][::-1],
                   color=palette, edgecolor='#0d1117', height=0.65)
    ax.axvline(global_med, color='white', linewidth=1.5, linestyle='--',
               label=f'Median: ${global_med:.0f}/hr')
    ax.set_xlabel('Median Hourly Rate (USD)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.spines[['top','right']].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#30363d')
    ax.legend(fontsize=8, labelcolor='#e6edf3', facecolor='#161b22')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.dataframe(country_data.head(n_show).rename(
        columns={'median': 'Median Rate (USD)', 'count': 'Job Count'}))

# ═══════════════════════ PAGE: TRENDS ══════════════════════════
elif page == "📈 Trends & Forecast":
    st.title("📈 Trends & Future Forecast")

    weekly_cat = (
        df.groupby(['week', 'category'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    top8 = df['category'].value_counts().head(8).index.tolist()
    weekly_top8 = weekly_cat[[c for c in top8 if c in weekly_cat.columns]]
    if len(weekly_top8.index) > 2:
        week_index = weekly_top8.index
        mid_weeks = (
            (week_index >= week_index.min() + 1)
            & (week_index <= week_index.max() - 1)
        )
        weekly_top8 = weekly_top8.loc[mid_weeks]


    st.subheader("Weekly Stacked Postings by Category")
    fig, ax = plt.subplots(figsize=(13, 6), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    colors = sns.color_palette('tab10', n_colors=weekly_top8.shape[1])
    ax.stackplot(weekly_top8.index, weekly_top8.T.values,
                 labels=weekly_top8.columns, colors=colors, alpha=0.88)
    ax.set_xlabel('ISO Week', color='#8b949e')
    ax.set_ylabel('Total Postings', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    ax.legend(loc='upper left', ncol=2, fontsize=8, labelcolor='#e6edf3', facecolor='#161b22')
    ax.spines[['top','right']].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Predicted Growth — Next Quarter")
    future = {
        'Blockchain / Crypto': 25, 'Data Science / AI': 18, 'DevOps / Cloud': 12,
        'Data Analysis': 10, 'Mobile Development': 8, 'Web Development': 5,
        'Marketing / SEO': 3, 'Finance / Accounting': 2,
        'Writing / Content': -2, 'Video / Animation': -3,
        'Graphic Design': -5, 'Customer Support': -8,
    }
    pred_df = pd.DataFrame.from_dict(future, orient='index', columns=['growth'])
    pred_df = pred_df.sort_values('growth', ascending=True)
    bar_colors = ['#2ecc71' if v >= 8 else ('#f39c12' if v >= 0 else '#e74c3c')
                  for v in pred_df['growth']]
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    bars = ax.barh(pred_df.index, pred_df['growth'],
                   color=bar_colors, edgecolor='#0d1117', height=0.65)
    ax.axvline(0, color='white', linewidth=1)
    ax.set_xlabel('Predicted Growth (%)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.spines[['top','right']].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
