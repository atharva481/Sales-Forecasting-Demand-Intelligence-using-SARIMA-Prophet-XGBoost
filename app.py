import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📦", layout="wide")

# ── DATA LOADER ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']    = df['Order Date'].dt.year
    df['Month']   = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Ship Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

@st.cache_data
def get_monthly(df, category=None, region=None):
    d = df.copy()
    if category and category != 'All': d = d[d['Category'] == category]
    if region   and region   != 'All': d = d[d['Region']   == region]
    ms = d.groupby(d['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
    ms['Order Date'] = ms['Order Date'].dt.to_timestamp()
    return ms.set_index('Order Date').sort_index()

@st.cache_data
def get_weekly(df):
    ws = df.groupby(df['Order Date'].dt.to_period('W'))['Sales'].sum().reset_index()
    ws['Order Date'] = ws['Order Date'].dt.to_timestamp()
    return ws.set_index('Order Date').sort_index()

@st.cache_data
def run_forecast(monthly_sales_json, horizon=3):
    from xgboost import XGBRegressor
    ms = pd.read_json(monthly_sales_json, typ='series')
    ms.index = pd.to_datetime(ms.index)

    def make_features(series):
        df_f = pd.DataFrame({'y': series.values}, index=series.index)
        df_f['lag1']    = df_f['y'].shift(1)
        df_f['lag2']    = df_f['y'].shift(2)
        df_f['lag3']    = df_f['y'].shift(3)
        df_f['roll3']   = df_f['y'].shift(1).rolling(3).mean()
        df_f['month']   = df_f.index.month
        df_f['quarter'] = df_f.index.quarter
        df_f['season']  = df_f.index.month % 12 // 3
        return df_f.dropna()

    feat = make_features(ms)
    X, y = feat.drop('y', axis=1), feat['y']
    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    last_vals = list(ms.values[-3:])
    preds, future_idx = [], []
    for step in range(horizon):
        next_month = ms.index[-1] + pd.DateOffset(months=step+1)
        row = {'lag1': last_vals[-1], 'lag2': last_vals[-2], 'lag3': last_vals[-3],
               'roll3': np.mean(last_vals[-3:]),
               'month': next_month.month, 'quarter': next_month.quarter,
               'season': next_month.month % 12 // 3}
        p = model.predict(pd.DataFrame([row]))[0]
        preds.append(p); last_vals.append(p); future_idx.append(next_month)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    if len(feat) >= 3:
        cv_pred = model.predict(X.iloc[-3:])
        mae  = mean_absolute_error(y.iloc[-3:], cv_pred)
        rmse = np.sqrt(mean_squared_error(y.iloc[-3:], cv_pred))
    else:
        mae = rmse = 0.0

    return preds, future_idx, round(mae), round(rmse)

@st.cache_data
def run_anomaly(weekly_json):
    from sklearn.ensemble import IsolationForest
    ws = pd.read_json(weekly_json, typ='series')
    ws.index = pd.to_datetime(ws.index)
    wdf = ws.reset_index()
    wdf.columns = ['date','sales']
    iso = IsolationForest(contamination=0.05, random_state=42)
    wdf['iso_anomaly'] = iso.fit_predict(wdf[['sales']]) == -1
    wdf['rolling_mean'] = wdf['sales'].rolling(4, min_periods=1).mean()
    wdf['rolling_std']  = wdf['sales'].rolling(4, min_periods=1).std().fillna(wdf['sales'].std())
    wdf['zscore']       = np.abs((wdf['sales'] - wdf['rolling_mean']) / wdf['rolling_std'])
    wdf['zscore_anomaly'] = wdf['zscore'] > 2
    return wdf

@st.cache_data
def run_clustering(df_json):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    df = pd.read_json(df_json)
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    sub_agg = df.groupby('Sub-Category').agg(
        total_sales=('Sales','sum'), avg_order=('Sales','mean'),
        volatility=('Sales','std'), n_orders=('Sales','count')
    ).reset_index()
    yr_sub = df.groupby(['Year','Sub-Category'])['Sales'].sum().reset_index()
    yrs = sorted(df['Year'].unique())
    if len(yrs) >= 2:
        y0 = yr_sub[yr_sub['Year']==yrs[0]].set_index('Sub-Category')['Sales']
        yn = yr_sub[yr_sub['Year']==yrs[-1]].set_index('Sub-Category')['Sales']
        growth = ((yn - y0) / y0 * 100).reset_index()
        growth.columns = ['Sub-Category','growth_rate']
        sub_agg = sub_agg.merge(growth, on='Sub-Category', how='left').fillna(0)
    else:
        sub_agg['growth_rate'] = 0

    feat_cols = ['total_sales','avg_order','volatility','growth_rate']
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(sub_agg[feat_cols])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    sub_agg['cluster'] = km.fit_predict(X_sc)
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feat_cols)
    order   = centers['total_sales'].rank(ascending=False).astype(int) - 1
    labels  = ['High Volume, Stable','Growing Demand','Low Volume, Volatile','Declining / Slow']
    sub_agg['Demand Cluster'] = sub_agg['cluster'].map({old: labels[new] for old, new in order.items()})
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)
    sub_agg['pca1'] = X_pca[:,0]; sub_agg['pca2'] = X_pca[:,1]
    return sub_agg

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("📦 Sales Forecasting")
st.sidebar.markdown("Internship Project — Week 3 & 4")
page = st.sidebar.radio("Navigate", [
    "📊 Sales Overview",
    "🔮 Forecast Explorer",
    "🚨 Anomaly Report",
    "🔵 Product Segments"
])

df = load_data()

# ── PAGE 1: SALES OVERVIEW ────────────────────────────────────────────────────
if page == "📊 Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    st.markdown("**Superstore Sales — 2020 to 2023 | King County, USA**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue",    f"${df['Sales'].sum()/1e6:.2f}M")
    col2.metric("Total Orders",     f"{len(df):,}")
    col3.metric("Avg Order Value",  f"${df['Sales'].mean():,.0f}")
    col4.metric("Avg Ship Days",    f"{df['Ship Days'].mean():.1f} days")

    st.divider()

    # Filters
    fc1, fc2 = st.columns(2)
    cat_filter = fc1.selectbox("Filter by Category", ['All'] + sorted(df['Category'].unique()))
    reg_filter = fc2.selectbox("Filter by Region",   ['All'] + sorted(df['Region'].unique()))

    ms = get_monthly(df, cat_filter, reg_filter)

    # Monthly trend
    st.subheader("Monthly Sales Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ms.index, ms['Sales']/1000, color='#2563EB', linewidth=2, marker='o', markersize=3)
    ax.fill_between(ms.index, ms['Sales']/1000, alpha=0.12, color='#2563EB')
    ax.set_ylabel('Sales ($K)'); ax.grid(axis='y', alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Bar + Pie
    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Revenue by Year")
        yr_sales = df.groupby('Year')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(yr_sales.index.astype(str), yr_sales.values/1e6,
                      color=['#0369A1','#0EA5E9','#38BDF8','#7DD3FC'])
        for bar, val in zip(bars, yr_sales.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'${val/1e6:.2f}M', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('Revenue ($M)'); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with b2:
        st.subheader("Sales by Category")
        cat_s = df.groupby('Category')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(cat_s.values, labels=cat_s.index, autopct='%1.1f%%',
               colors=['#2563EB','#16A34A','#D97706'], startangle=140)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Sales by Region")
    reg_yr = df.groupby(['Year','Region'])['Sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    for reg, grp in reg_yr.groupby('Region'):
        ax.plot(grp['Year'], grp['Sales']/1e6, marker='o', linewidth=2.2, label=reg)
    ax.set_ylabel('Sales ($M)'); ax.legend(); ax.grid(alpha=0.35)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 2: FORECAST EXPLORER ─────────────────────────────────────────────────
elif page == "🔮 Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("**XGBoost model — lag features + seasonality**")

    fc1, fc2, fc3 = st.columns(3)
    seg_type = fc1.selectbox("Segment Type", ["Overall", "Category", "Region"])
    seg_val  = "All"
    if seg_type == "Category":
        seg_val = fc2.selectbox("Category", sorted(df['Category'].unique()))
    elif seg_type == "Region":
        seg_val = fc2.selectbox("Region", sorted(df['Region'].unique()))
    horizon = fc3.slider("Forecast Horizon (months)", 1, 3, 3)

    cat_arg = seg_val if seg_type == "Category" else None
    reg_arg = seg_val if seg_type == "Region"   else None
    ms = get_monthly(df, cat_arg, reg_arg)

    preds, future_idx, mae, rmse = run_forecast(ms['Sales'].to_json(), horizon)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ms.index, ms['Sales']/1000, color='#94A3B8', linewidth=1.8, label='Historical')
    ax.plot(future_idx, np.array(preds)/1000, color='#DC2626', linewidth=2.5,
            marker='o', linestyle='--', label='Forecast')
    for i, (idx, pred) in enumerate(zip(future_idx, preds)):
        ax.annotate(f'${pred/1000:.1f}K', (idx, pred/1000),
                    textcoords='offset points', xytext=(0,10), ha='center', fontsize=9, color='#DC2626')
    ax.set_ylabel('Sales ($K)'); ax.legend(); ax.grid(alpha=0.35)
    ax.set_title(f'{seg_type} {"("+seg_val+")" if seg_val != "All" else ""} — {horizon}-Month Forecast',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    m1, m2 = st.columns(2)
    m1.metric("Model MAE (hold-out)",  f"${mae:,}")
    m2.metric("Model RMSE (hold-out)", f"${rmse:,}")

    fc_df = pd.DataFrame({'Month': [f'M+{i+1}' for i in range(horizon)],
                          'Forecast ($)': [f'${v:,.0f}' for v in preds]})
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

# ── PAGE 3: ANOMALY REPORT ────────────────────────────────────────────────────
elif page == "🚨 Anomaly Report":
    st.title("🚨 Anomaly Detection Report")

    ws = get_weekly(df)
    wdf = run_anomaly(ws['Sales'].to_json())

    method = st.radio("Detection Method", ["Isolation Forest", "Z-Score (rolling)", "Both"], horizontal=True)
    col_map = {"Isolation Forest": "iso_anomaly", "Z-Score (rolling)": "zscore_anomaly"}

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wdf['date'], wdf['sales']/1000, color='#94A3B8', linewidth=1.2, alpha=0.8, label='Weekly Sales')

    if method == "Both":
        for col, color, label in [('iso_anomaly','#DC2626','Isolation Forest'),
                                    ('zscore_anomaly','#D97706','Z-Score')]:
            anom = wdf[wdf[col]]
            ax.scatter(anom['date'], anom['sales']/1000, color=color, s=80,
                       zorder=5, marker='^', label=f'{label} ({len(anom)})')
    else:
        col   = col_map[method]
        color = '#DC2626' if method == "Isolation Forest" else '#D97706'
        anom  = wdf[wdf[col]]
        ax.scatter(anom['date'], anom['sales']/1000, color=color, s=90,
                   zorder=5, marker='^', label=f'Anomaly ({len(anom)})')

    ax.set_title('Weekly Sales — Anomaly Detection', fontsize=13, fontweight='bold')
    ax.set_ylabel('Weekly Sales ($K)'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    a1, a2, a3 = st.columns(3)
    a1.metric("Isolation Forest anomalies", int(wdf['iso_anomaly'].sum()))
    a2.metric("Z-Score anomalies",          int(wdf['zscore_anomaly'].sum()))
    a3.metric("Both agree",                 int((wdf['iso_anomaly'] & wdf['zscore_anomaly']).sum()))

    st.subheader("Detected Anomaly Dates")
    show_col = 'iso_anomaly' if method != "Z-Score (rolling)" else 'zscore_anomaly'
    anom_table = wdf[wdf[show_col]][['date','sales','zscore']].copy()
    anom_table.columns = ['Week Start','Sales ($)','Z-Score']
    anom_table['Sales ($)'] = anom_table['Sales ($)'].apply(lambda x: f'${x:,.0f}')
    anom_table['Z-Score']   = anom_table['Z-Score'].round(2)
    st.dataframe(anom_table, use_container_width=True, hide_index=True)

    st.info("📌 **Interpretation:** Isolation Forest detects global outliers based on the full distribution. "
            "Z-Score flags sudden local deviations vs. recent rolling average. "
            "Use both together for complete anomaly coverage.")

# ── PAGE 4: PRODUCT SEGMENTS ──────────────────────────────────────────────────
elif page == "🔵 Product Segments":
    st.title("🔵 Product Demand Segments")
    st.markdown("K-Means clustering on sub-category sales features (volume, volatility, growth, avg order)")

    df_json = df[['Order Date','Sales','Sub-Category','Year']].assign(
        **{'Order Date': df['Order Date'].astype(str)}).to_json()
    sub_agg = run_clustering(df_json)

    palette = {'High Volume, Stable':'#2563EB','Growing Demand':'#16A34A',
               'Low Volume, Volatile':'#D97706','Declining / Slow':'#DC2626'}

    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl, grp in sub_agg.groupby('Demand Cluster'):
        ax.scatter(grp['pca1'], grp['pca2'], label=lbl, s=120,
                   color=palette.get(lbl,'gray'), edgecolors='white', lw=0.5)
        for _, row in grp.iterrows():
            ax.annotate(row['Sub-Category'], (row['pca1'], row['pca2']),
                        fontsize=8.5, ha='center', va='bottom')
    ax.set_title('Sub-Category Demand Clusters (PCA)', fontsize=13, fontweight='bold')
    ax.set_xlabel('PC1 (Sales Volume)'); ax.set_ylabel('PC2 (Volatility / Growth)')
    ax.legend(fontsize=9); plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Cluster Membership")
    table = sub_agg[['Sub-Category','Demand Cluster','total_sales','growth_rate']].copy()
    table.columns = ['Sub-Category','Demand Cluster','Total Sales ($)','YoY Growth (%)']
    table['Total Sales ($)'] = table['Total Sales ($)'].apply(lambda x: f'${x:,.0f}')
    table['YoY Growth (%)']  = table['YoY Growth (%)'].round(1)
    st.dataframe(table.sort_values('Demand Cluster'), use_container_width=True, hide_index=True)

    st.subheader("Recommended Stocking Strategy")
    strategies = {
        "🔵 High Volume, Stable":    "Safety stock = 15% above forecast. Auto-replenish. Never go out of stock.",
        "🟢 Growing Demand":         "Increase order qty 20–25% QoQ. Lock in bulk pricing with suppliers proactively.",
        "🟡 Low Volume, Volatile":   "Just-in-time ordering. Minimal stock. Review quarterly for discontinuation.",
        "🔴 Declining / Slow":       "Run clearance promotions. Cut reorder qty by 30%. Review for delisting in 2 quarters.",
    }
    for label, strategy in strategies.items():
        st.markdown(f"**{label}** — {strategy}")
